"""Retired on 2026-09-30: the per-member maps of the tc evaluator.

This file is not importable and is not run. It holds, verbatim, the code that was
removed from the live package when the maps were retired, so that its logic stays
readable. See ../README.md (eval/_quarantine/20260930/README.md) for why and what
replaces it.

Sections, each copied unchanged from commit b83cca2:
  1. run_member_maps         from eval/evaluators/tc/core/workflows.py (lines 538-624)
  2. load_prediction_member_fields
                             from eval/evaluators/tc/core/loading_predictions.py (lines 118-250)
  3. the runner block        from eval/evaluators/tc/runner.py (lines 492-551), the end of run()
  4. the member-maps subcommand
                             from main() in eval/evaluators/tc/core/workflows.py
                             (lines 856-866 and 917-929)
The page itself was drawn by _plot_member_page in member_plot.py, next to this file.
"""

# ---------------------------------------------------------------------------
# 1. eval/evaluators/tc/core/workflows.py
# ---------------------------------------------------------------------------

def run_member_maps(
    *,
    predictions_dir: str,
    outdir: str,
    run_label: str,
    display_label: str | None = None,
    event_names: list[str] | None = None,
    date: str,
    steps: list[int] | None = None,
    members: list[int] | None = None,
) -> list[str]:
    """Member spatial maps workflow."""
    display_label = display_label or run_label
    steps = steps or [24, 120]
    members = members or [0, 1, 2, 3, 4]

    pred_dir = Path(predictions_dir).expanduser().resolve()
    out_dir = Path(outdir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = discover_prediction_files(pred_dir)
    if not pred_files:
        raise FileNotFoundError(f"No predictions_*.nc files found in {pred_dir}")

    selected_events = event_names or list(EVENTS.keys())
    generated: list[str] = []

    for event_name in selected_events:
        if event_name not in EVENTS:
            LOG.warning("Unknown event=%s, skipping", event_name)
            continue
        event = EVENTS[event_name]
        exp_cfg = EXPERIMENT_CONFIGS.get(event_name)
        plot_cfg = PLOT_CONFIGS.get(event_name, TCPlotConfig())

        event_pred_files = select_prediction_files_for_event(pred_files, event)
        if not event_pred_files:
            LOG.info("Skipping event=%s: no matching prediction files", event_name)
            continue

        date_int = int(date)
        step_files = [
            (p, ymd, step) for p, ymd, step in event_pred_files
            if ymd == date_int and step in steps
        ]
        if not step_files:
            LOG.warning(
                "Skipping event=%s: no prediction files for date=%s steps=%s",
                event_name, date, steps,
            )
            continue

        safe_label = display_label.replace(" ", "_").replace("/", "_")
        pdf_name = f"tc_members_{event_name}_{safe_label}_{date}.pdf"
        pdf_path = out_dir / pdf_name

        with FigureBook(pdf_path, png=True) as book:
            for nc_path, ymd, step in sorted(step_files, key=lambda x: x[2]):
                LOG.info("Loading event=%s date=%s step=%d", event_name, date, step)
                try:
                    fields = load_prediction_member_fields(
                        nc_path, event.bbox, members,
                        regrid_resolution=plot_cfg.regrid_resolution,
                    )
                except Exception as exc:
                    LOG.warning("Failed to load %s: %s", nc_path.name, exc)
                    continue
                for mi, mbr in enumerate(members):
                    LOG.info("  Plotting member %d (step=%dh)", mbr, step)
                    fig = _plot_member_page(
                        fields,
                        bbox=event.bbox,
                        plot_config=plot_cfg,
                        exp_config=exp_cfg,
                        member_idx=mi,
                        member_label=mbr,
                        step_hours=step,
                        date_str=date,
                        display_label=display_label,
                        event_name=event_name,
                    )
                    book.add(fig, name=f"step{step:03d}_member{mbr}")

        LOG.info("Saved TC member maps PDF: %s", pdf_path)
        generated.append(str(pdf_path))

    return generated


# ---------------------------------------------------------------------------
# 2. eval/evaluators/tc/core/loading_predictions.py
# ---------------------------------------------------------------------------

def load_prediction_member_fields(
    nc_path: Path,
    bbox: BoundingBox,
    members: list[int],
    regrid_resolution: float = 0.25,
) -> dict[str, np.ndarray]:
    """Load input, prediction, and truth fields for selected members, masked to event bbox.

    Returns dict with keys: x_interp_msl, x_interp_wind, y_pred_msl, y_pred_wind,
    y_msl, y_wind, lat_axis, lon_axis.
    Arrays are shaped [len(members), nlat, nlon].
    """
    from scipy.interpolate import griddata

    with xr.open_dataset(nc_path) as ds:
        weather_states = ds["weather_state"].values.tolist()
        i_msl = weather_states.index("msl")
        i_u10 = weather_states.index("10u")
        i_v10 = weather_states.index("10v")

        lon_flat, lat_flat = prediction_point_coordinates(ds)

        y_pred_raw = np.asarray(ds["y_pred"].isel(sample=0).values, dtype=np.float64)[members]
        y_raw = np.asarray(ds["y"].isel(sample=0).values, dtype=np.float64)[members]

        if "x_interp" in ds:
            x_interp_raw = np.asarray(ds["x_interp"].isel(sample=0).values, dtype=np.float64)[members]
            x_lres_raw = None
            x_lon_lres = None
            x_lat_lres = None
        elif "x" in ds:
            x_interp_raw = None
            x_lres_raw = np.asarray(ds["x"].isel(sample=0).values, dtype=np.float64)[members]
            lon_lres_da = ds["lon_lres"]
            lat_lres_da = ds["lat_lres"]
            if lon_lres_da.ndim == 2:
                lon_lres_da = lon_lres_da.isel({lon_lres_da.dims[0]: 0})
                lat_lres_da = lat_lres_da.isel({lat_lres_da.dims[0]: 0})
            x_lon_lres = normalize_lon(np.asarray(lon_lres_da.values, dtype=np.float64))
            x_lat_lres = np.asarray(lat_lres_da.values, dtype=np.float64)
        else:
            x_interp_raw = None
            x_lres_raw = None
            x_lon_lres = None
            x_lat_lres = None

    mask = point_mask(lon_flat, lat_flat, bbox)
    if not np.any(mask):
        raise RuntimeError(f"No grid points inside event bbox")

    lon_bbox = lon_flat[mask]
    lat_bbox = lat_flat[mask]
    y_pred_bbox = y_pred_raw[:, mask, :]
    y_bbox = y_raw[:, mask, :]
    x_interp_bbox = x_interp_raw[:, mask, :] if x_interp_raw is not None else None

    if x_lres_raw is not None:
        lres_mask = point_mask(x_lon_lres, x_lat_lres, bbox)
        if np.any(lres_mask):
            x_lres_bbox = x_lres_raw[:, lres_mask, :]
            x_lon_bbox = x_lon_lres[lres_mask]
            x_lat_bbox = x_lat_lres[lres_mask]
        else:
            x_lres_bbox = None
    else:
        x_lres_bbox = None

    # Build regular target grid within bbox
    west = normalize_lon(np.asarray([bbox.west], dtype=np.float64))[0]
    east = normalize_lon(np.asarray([bbox.east], dtype=np.float64))[0]
    lon_axis = np.arange(west, east + regrid_resolution / 4, regrid_resolution)
    lat_axis = np.arange(bbox.south, bbox.north + regrid_resolution / 4, regrid_resolution)
    target_lon, target_lat = np.meshgrid(lon_axis, lat_axis)

    def _interp_to_grid(flat_vals: np.ndarray) -> np.ndarray:
        return griddata(
            (lon_bbox, lat_bbox),
            flat_vals,
            (target_lon, target_lat),
            method="linear",
        )

    def _fields_for(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        msl_list, wind_list = [], []
        for mi in range(raw.shape[0]):
            msl_grid = _interp_to_grid(raw[mi, :, i_msl] / 100.0)
            u10_grid = _interp_to_grid(raw[mi, :, i_u10])
            v10_grid = _interp_to_grid(raw[mi, :, i_v10])
            wind_grid = np.sqrt(u10_grid ** 2 + v10_grid ** 2)
            msl_list.append(msl_grid)
            wind_list.append(wind_grid)
        return np.array(msl_list), np.array(wind_list)

    pred_msl, pred_wind = _fields_for(y_pred_bbox)
    truth_msl, truth_wind = _fields_for(y_bbox)

    result = {
        "y_pred_msl": pred_msl,
        "y_pred_wind": pred_wind,
        "y_msl": truth_msl,
        "y_wind": truth_wind,
        "lat_axis": lat_axis,
        "lon_axis": lon_axis,
    }

    if x_interp_bbox is not None:
        input_msl, input_wind = _fields_for(x_interp_bbox)
        result["x_interp_msl"] = input_msl
        result["x_interp_wind"] = input_wind
    elif x_lres_bbox is not None:
        def _interp_lres(flat_vals: np.ndarray) -> np.ndarray:
            return griddata(
                (x_lon_bbox, x_lat_bbox),
                flat_vals,
                (target_lon, target_lat),
                method="linear",
            )

        def _fields_lres(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            msl_list, wind_list = [], []
            for mi in range(raw.shape[0]):
                msl_grid = _interp_lres(raw[mi, :, i_msl] / 100.0)
                u10_grid = _interp_lres(raw[mi, :, i_u10])
                v10_grid = _interp_lres(raw[mi, :, i_v10])
                wind_grid = np.sqrt(u10_grid ** 2 + v10_grid ** 2)
                msl_list.append(msl_grid)
                wind_list.append(wind_grid)
            return np.array(msl_list), np.array(wind_list)

        input_msl, input_wind = _fields_lres(x_lres_bbox)
        result["x_interp_msl"] = input_msl
        result["x_interp_wind"] = input_wind



# ---------------------------------------------------------------------------
# 3. eval/evaluators/tc/runner.py, end of run() (inside the function body)
# ---------------------------------------------------------------------------

def _runner_member_maps_block(eval_config, output_dir, predictions_dir, run_label, event_names):
    # Member maps (optional, controlled by eval_config["member_maps"])
    mm_cfg = eval_config.get("member_maps") or {}
    if mm_cfg.get("enabled"):
        mm_steps = mm_cfg.get("steps") or [24, 120]
        mm_members = mm_cfg.get("members") or list(range(10))
        mm_outdir = output_dir / "member_maps"

        # Per-event dates (preferred) or global dates (legacy)
        event_dates = mm_cfg.get("event_dates") or {}
        if event_dates:
            for evt, date in event_dates.items():
                try:
                    run_member_maps(
                        predictions_dir=str(predictions_dir),
                        outdir=str(mm_outdir),
                        run_label=run_label,
                        event_names=[evt],
                        date=str(date),
                        steps=mm_steps,
                        members=mm_members,
                    )
                    LOG.info("Member maps written for event=%s date=%s", evt, date)
                except Exception:
                    LOG.error("Member maps failed for event=%s date=%s", evt, date, exc_info=True)
        else:
            mm_dates = mm_cfg.get("dates") or []
            mm_events = mm_cfg.get("events") or list(event_names)
            for date in mm_dates:
                try:
                    run_member_maps(
                        predictions_dir=str(predictions_dir),
                        outdir=str(mm_outdir),
                        run_label=run_label,
                        event_names=mm_events,
                        date=date,
                        steps=mm_steps,
                        members=mm_members,
                    )
                    LOG.info("Member maps written for date=%s", date)
                except Exception:
                    LOG.error("Member maps failed for date=%s", date, exc_info=True)

        # Combined PDF: merge all individual member map PDFs into one
        if mm_cfg.get("combined_pdf") and mm_outdir.exists():
            individual_pdfs = sorted(mm_outdir.glob("tc_members_*.pdf"))
            if len(individual_pdfs) > 1:
                try:
                    from pypdf import PdfReader, PdfWriter
                    writer = PdfWriter()
                    for pdf_path in individual_pdfs:
                        for page in PdfReader(str(pdf_path)).pages:
                            writer.add_page(page)
                    combined_name = "_".join(event_dates.keys()) if event_dates else "combined"
                    safe_label = run_label.replace(" ", "_").replace("/", "_")
                    combined_path = mm_outdir / f"tc_members_{combined_name}_{safe_label}.pdf"
                    with open(combined_path, "wb") as f:
                        writer.write(f)
                    LOG.info("Combined member maps PDF: %s", combined_path)
                except Exception:
                    LOG.error("Failed to merge member map PDFs", exc_info=True)


# ---------------------------------------------------------------------------
# 4. eval/evaluators/tc/core/workflows.py, main(): the member-maps subcommand
# ---------------------------------------------------------------------------

def _main_member_maps_excerpt(subparsers, args):
    # --- member-maps subcommand ---
    mm_parser = subparsers.add_parser("member-maps", help="Generate TC member spatial maps.")
    mm_parser.add_argument("--predictions-dir", required=True)
    mm_parser.add_argument("--outdir", required=True)
    mm_parser.add_argument("--run-label", required=True)
    mm_parser.add_argument("--display-label", default="")
    mm_parser.add_argument("--events", default="")
    mm_parser.add_argument("--date", required=True)
    mm_parser.add_argument("--steps", default="24,120")
    mm_parser.add_argument("--members", default="0,1,2,3,4")
    mm_parser.add_argument("--log-level", default="INFO")
    if args.command == "member-maps":
        event_names = [v.strip() for v in args.events.split(",") if v.strip()] or None
        steps = [int(s.strip()) for s in args.steps.split(",") if s.strip()]
        run_member_maps(
            predictions_dir=args.predictions_dir,
            outdir=args.outdir,
            run_label=args.run_label,
            display_label=args.display_label or None,
            event_names=event_names,
            date=args.date,
            steps=steps,
            members=_parse_members(args.members),
        )
