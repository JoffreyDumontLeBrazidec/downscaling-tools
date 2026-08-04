"""Reusable ECMWF tctracker production, verification, and parsing helpers."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import tarfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

BASINS = ("atl", "enp", "cnp", "wnp", "nin", "sin", "aus", "spc")
KT_TO_MS = 1.0 / 1.94384


@dataclass(frozen=True)
class TCTrackerTarget:
    date: str
    member: int
    tag: str
    tar_path: Path
    log_path: Path
    status_path: Path
    contents_path: Path
    sha256_path: Path


@dataclass(frozen=True)
class TCTrackerConfig:
    lane: str
    expver: str
    output_dir: Path
    dates: tuple[str, ...]
    members: tuple[int, ...]
    time: str = "00"
    start_step: int = 0
    end_step: int = 360
    step_interval: int = 24
    grid: int = 320
    fdb_class: str = "rd"
    fdb_type: str = "pf"
    stream: str = "enfo"
    vorticity: bool = False
    model_keyword: str = ""
    module: str = "tctracker"
    overwrite: bool = False

    @property
    def tars_dir(self) -> Path:
        return self.output_dir / "tars"

    @property
    def logs_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def manifests_dir(self) -> Path:
        return self.output_dir / "manifests"

    @property
    def targets(self) -> list[TCTrackerTarget]:
        targets: list[TCTrackerTarget] = []
        for date in self.dates:
            for member in self.members:
                mem3 = f"{member:03d}"
                tag = f"{self.expver}_{date}_{self.time}_m{mem3}_o{self.grid}_tracks"
                targets.append(TCTrackerTarget(
                    date=date,
                    member=member,
                    tag=tag,
                    tar_path=self.tars_dir / f"{tag}.tar",
                    log_path=self.logs_dir / f"{tag}.log",
                    status_path=self.manifests_dir / f"{tag}.status",
                    contents_path=self.manifests_dir / f"{tag}.contents",
                    sha256_path=self.manifests_dir / f"{tag}.sha256",
                ))
        return targets


def _csv_ints(raw: str | Iterable[int] | None, default: Iterable[int]) -> tuple[int, ...]:
    if raw is None:
        return tuple(int(v) for v in default)
    if isinstance(raw, str):
        return tuple(int(tok.strip()) for tok in raw.split(",") if tok.strip())
    return tuple(int(v) for v in raw)


def _csv_strs(raw: str | Iterable[str] | None, default: Iterable[str]) -> tuple[str, ...]:
    if raw is None:
        return tuple(str(v) for v in default)
    if isinstance(raw, str):
        return tuple(tok.strip() for tok in raw.split(",") if tok.strip())
    return tuple(str(v) for v in raw)


def _lane_short_name(lane: str, lane_config: dict[str, Any]) -> str:
    configured = (lane_config.get("tctracker") or {}).get("output_lane")
    if configured:
        return str(configured)
    match = re.search(r"o\d+_o\d+", lane)
    return match.group(0) if match else lane


def build_config(args: Any, lane_config: dict[str, Any], host_config: dict[str, Any], output_dir: Path) -> TCTrackerConfig:
    cfg = lane_config.get("tctracker") or {}
    predict = lane_config.get("predict") or {}
    expver = getattr(args, "expver", None) or cfg.get("expver")
    if not expver:
        raise SystemExit("tctracker requires --expver or lane_config['tctracker']['expver']")

    dates = _csv_strs(getattr(args, "dates", None), cfg.get("dates") or predict.get("dates") or ())
    members = _csv_ints(getattr(args, "members", None), cfg.get("members") or predict.get("members") or ())
    if not dates:
        raise SystemExit("tctracker needs dates from --dates, tctracker.dates, or predict.dates")
    if not members:
        raise SystemExit("tctracker needs members from --members, tctracker.members, or predict.members")

    vorticity_raw = getattr(args, "vorticity", None)
    if vorticity_raw is None:
        vorticity = bool(cfg.get("vorticity", False))
    else:
        vorticity = str(vorticity_raw).lower() in {"1", "true", "yes", "y"}

    explicit_output = getattr(args, "output_dir", None)
    if explicit_output:
        out = Path(explicit_output)
    else:
        out = output_dir

    return TCTrackerConfig(
        lane=getattr(args, "lane", ""),
        expver=str(expver),
        output_dir=out,
        dates=dates,
        members=members,
        time=str(getattr(args, "time", None) or cfg.get("time", "00")),
        start_step=int(getattr(args, "start_step", None) if getattr(args, "start_step", None) is not None else cfg.get("start_step", 0)),
        end_step=int(getattr(args, "end_step", None) if getattr(args, "end_step", None) is not None else cfg.get("end_step", 360)),
        step_interval=int(getattr(args, "step_interval", None) if getattr(args, "step_interval", None) is not None else cfg.get("step_interval", 24)),
        grid=int(getattr(args, "grid", None) if getattr(args, "grid", None) is not None else cfg.get("grid", 320)),
        fdb_class=str(getattr(args, "fdb_class", None) or cfg.get("class", cfg.get("fdb_class", "rd"))),
        fdb_type=str(getattr(args, "fdb_type", None) or cfg.get("type", cfg.get("fdb_type", "pf"))),
        stream=str(getattr(args, "stream", None) or cfg.get("stream", "enfo")),
        vorticity=vorticity,
        model_keyword=str(getattr(args, "model_keyword", None) if getattr(args, "model_keyword", None) is not None else cfg.get("model_keyword", "")),
        module=str(getattr(args, "module", None) or cfg.get("module", "tctracker")),
        overwrite=bool(getattr(args, "overwrite", False)),
    )


def default_output_dir(host_config: dict[str, Any], lane: str, lane_config: dict[str, Any], expver: str) -> Path:
    return Path(host_config["scratch_root"]) / "eval" / _lane_short_name(lane, lane_config) / "tctracker" / expver


def build_tctracker_command(config: TCTrackerConfig, target: TCTrackerTarget) -> list[str]:
    return [
        "tctracker",
        "-v", str(config.vorticity).lower(),
        "-g", "true",
        "-r", str(config.grid),
        "-C", config.fdb_class,
        "-T", config.fdb_type,
        "-S", config.stream,
        "-E", config.expver,
        "-N", str(target.member),
        "-d", target.date,
        "-t", config.time,
        "-s", str(config.start_step),
        "-f", str(config.end_step),
        "-i", str(config.step_interval),
        "-o", str(target.tar_path),
    ]


def _ensure_dirs(config: TCTrackerConfig) -> None:
    for path in (config.tars_dir, config.logs_dir, config.manifests_dir):
        path.mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tar_contents(path: Path) -> list[str]:
    with tarfile.open(path) as tf:
        return sorted(tf.getnames())


def _write_status(target: TCTrackerTarget, fields: dict[str, Any]) -> None:
    payload = {
        "tag": target.tag,
        "date": target.date,
        "member": target.member,
        **fields,
        "tar": target.tar_path,
        "log": target.log_path,
    }
    target.status_path.write_text("".join(f"{k}={v}\n" for k, v in payload.items()), encoding="utf-8")


def _write_manifest(config: TCTrackerConfig) -> Path:
    path = config.manifests_dir / "batch_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tag", "date", "member", "status", "rc", "tar", "bytes", "sha256"])
        for target in config.targets:
            status = "missing"
            rc = ""
            if target.status_path.exists():
                data = _read_status(target.status_path)
                status = data.get("status", status)
                rc = data.get("rc", rc)
            bytes_ = target.tar_path.stat().st_size if target.tar_path.exists() else 0
            sha = target.sha256_path.read_text().split()[0] if target.sha256_path.exists() else ""
            writer.writerow([target.tag, target.date, target.member, status, rc, target.tar_path, bytes_, sha])
    return path


def _read_status(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            data[key] = value
    return data


def _record_tar_artifacts(target: TCTrackerTarget) -> None:
    contents = _tar_contents(target.tar_path)
    target.contents_path.write_text("\n".join(contents) + "\n", encoding="utf-8")
    target.sha256_path.write_text(f"{_sha256(target.tar_path)}  {target.tar_path}\n", encoding="utf-8")


def run_one(config: TCTrackerConfig, target: TCTrackerTarget) -> str:
    start = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if target.tar_path.exists() and target.tar_path.stat().st_size > 0 and not config.overwrite:
        _record_tar_artifacts(target)
        _write_status(target, {
            "start_utc": start,
            "end_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": "skipped_existing",
            "rc": 0,
        })
        return "skipped_existing"

    cmd = build_tctracker_command(config, target)
    target.log_path.parent.mkdir(parents=True, exist_ok=True)
    shell = " && ".join([
        f"module load {shlex.quote(config.module)} >/dev/null 2>&1",
        f"export model_keyword={shlex.quote(config.model_keyword)}",
        " ".join(shlex.quote(part) for part in cmd),
    ])
    with target.log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(["bash", "-lc", shell], stdout=log, stderr=subprocess.STDOUT, check=False)
    end = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ok = proc.returncode == 0 and target.tar_path.exists() and target.tar_path.stat().st_size > 0
    if ok:
        _record_tar_artifacts(target)
    _write_status(target, {
        "start_utc": start,
        "end_utc": end,
        "status": "done" if ok else "failed",
        "rc": proc.returncode,
    })
    if not ok:
        raise RuntimeError(f"tctracker failed for {target.tag} rc={proc.returncode}; see {target.log_path}")
    return "done"


def run_batch(config: TCTrackerConfig) -> Path:
    _ensure_dirs(config)
    start = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (config.manifests_dir / "batch_status.txt").write_text(f"batch_start_utc={start}\nmode=eval_cli_tctracker\n", encoding="utf-8")
    failures: list[str] = []
    for target in config.targets:
        try:
            run_one(config, target)
        except Exception as exc:
            failures.append(f"{target.tag}: {exc}")
    manifest = _write_manifest(config)
    end = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (config.manifests_dir / "batch_status.txt").write_text(
        f"batch_start_utc={start}\nmode=eval_cli_tctracker\nbatch_end_utc={end}\noverall_rc={1 if failures else 0}\n"
        f"tar_count={len(list(config.tars_dir.glob('*.tar')))}\nstatus_count={len(list(config.manifests_dir.glob('*.status')))}\n",
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError("tctracker batch failures:\n" + "\n".join(failures))
    return manifest


def verify_outputs(config: TCTrackerConfig) -> dict[str, Any]:
    issues: list[str] = []
    status_counts: dict[str, int] = {}
    rc_counts: dict[str, int] = {}
    expected_basin_suffixes = {f"_{basin}" for basin in BASINS}

    for target in config.targets:
        if not target.tar_path.exists() or target.tar_path.stat().st_size == 0:
            issues.append(f"missing_or_empty_tar:{target.tar_path}")
        else:
            try:
                names = _tar_contents(target.tar_path)
                suffixes = {"_" + name.rsplit("_", 1)[-1] for name in names}
                missing = sorted(expected_basin_suffixes - suffixes)
                if missing:
                    issues.append(f"{target.tag}:missing_basin_files:{missing}")
            except Exception as exc:
                issues.append(f"{target.tag}:tar_read_error:{exc}")
        for path, label in ((target.status_path, "status"), (target.contents_path, "contents"), (target.sha256_path, "sha256")):
            if not path.exists() or path.stat().st_size == 0:
                issues.append(f"{target.tag}:missing_{label}:{path}")
        if target.status_path.exists():
            data = _read_status(target.status_path)
            status_counts[data.get("status", "unknown")] = status_counts.get(data.get("status", "unknown"), 0) + 1
            rc_counts[data.get("rc", "unknown")] = rc_counts.get(data.get("rc", "unknown"), 0) + 1

    manifest_path = config.manifests_dir / "batch_manifest.csv"
    manifest_rows = 0
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as fh:
            manifest_rows = max(0, sum(1 for _ in fh) - 1)
    else:
        issues.append(f"missing_batch_manifest:{manifest_path}")

    return {
        "expected": len(config.targets),
        "tar_count": sum(1 for t in config.targets if t.tar_path.exists() and t.tar_path.stat().st_size > 0),
        "status_count": sum(1 for t in config.targets if t.status_path.exists()),
        "contents_count": sum(1 for t in config.targets if t.contents_path.exists()),
        "sha256_count": sum(1 for t in config.targets if t.sha256_path.exists()),
        "manifest_rows": manifest_rows,
        "status_counts": status_counts,
        "rc_counts": rc_counts,
        "issues": issues,
    }


def write_verification_summary(config: TCTrackerConfig, verification: dict[str, Any]) -> tuple[Path, Path]:
    config.manifests_dir.mkdir(parents=True, exist_ok=True)
    json_path = config.manifests_dir / "verification_summary.json"
    md_path = config.manifests_dir / "verification_summary.md"
    json_path.write_text(json.dumps(verification, indent=2, default=str) + "\n", encoding="utf-8")
    md_path.write_text(
        "# tctracker verification summary\n\n"
        f"Generated UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n\n"
        "## Structural checks\n"
        f"- Tars: {verification['tar_count']}/{verification['expected']}\n"
        f"- Status files: {verification['status_count']}/{verification['expected']}\n"
        f"- Contents files: {verification['contents_count']}/{verification['expected']}\n"
        f"- SHA256 files: {verification['sha256_count']}/{verification['expected']}\n"
        f"- Batch manifest rows: {verification['manifest_rows']}/{verification['expected']}\n"
        f"- Status counts: {verification['status_counts']}\n"
        f"- RC counts: {verification['rc_counts']}\n"
        f"- Issues: {len(verification['issues'])}\n",
        encoding="utf-8",
    )
    return md_path, json_path


def _parse_atlantic_file(text: str, source: str) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        header = re.match(r"\d{5}\s+(\d+/\d+/\d+)\s+M=\s*(\d+)\s+(\d+)\s+SNBR=\s*(\d+)", line)
        if header:
            if current and current["records"]:
                tracks.append(current)
            current = {
                "source": source,
                "start_date": header.group(1),
                "n_steps": int(header.group(2)),
                "seq": int(header.group(3)),
                "snbr": int(header.group(4)),
                "records": [],
                "classification": None,
            }
            continue
        cls = re.match(r"\d+\s+(HR\d|TS|TD|ET|SSD)\s*$", line)
        if cls and current is not None:
            current["classification"] = cls.group(1)
            continue
        rec = re.match(r"\d{5}\s+(\d{4}/\d{2}/\d{2}/\d{2})\*(\d{7})\s+(\d+)\s+(\d+)\*", line)
        if rec and current is not None:
            raw = rec.group(2)
            current["records"].append({
                "valid_time": rec.group(1),
                "lat": int(raw[:3]) / 10.0,
                "lon_e": int(raw[3:]) / 10.0,
                "wind_kt": int(rec.group(3)),
                "wind_ms": int(rec.group(3)) * KT_TO_MS,
                "mslp_hpa": int(rec.group(4)),
            })
    if current and current["records"]:
        tracks.append(current)
    for track in tracks:
        records = track["records"]
        if records:
            track["wind_max_kt"] = max(r["wind_kt"] for r in records)
            track["wind_max_ms"] = max(r["wind_ms"] for r in records)
            track["mslp_min_hpa"] = min(r["mslp_hpa"] for r in records)
            track["first_valid_time"] = records[0]["valid_time"]
            track["last_valid_time"] = records[-1]["valid_time"]
    return tracks


def parse_atlantic_tracks(config: TCTrackerConfig) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []
    for target in config.targets:
        if not target.tar_path.exists():
            continue
        with tarfile.open(target.tar_path) as tf:
            atl_names = [name for name in tf.getnames() if name.endswith("_atl")]
            for name in atl_names:
                fh = tf.extractfile(name)
                if fh is None:
                    continue
                text = fh.read().decode("utf-8", errors="replace")
                for track in _parse_atlantic_file(text, target.tag):
                    track["init_date"] = target.date
                    track["member"] = target.member
                    tracks.append(track)
    return tracks


def write_atlantic_summary(config: TCTrackerConfig, tracks: list[dict[str, Any]]) -> tuple[Path, Path]:
    config.manifests_dir.mkdir(parents=True, exist_ok=True)
    json_path = config.manifests_dir / "atlantic_tracks_summary.json"
    md_path = config.manifests_dir / "atlantic_tracks_summary.md"
    wind_values = [float(t["wind_max_ms"]) for t in tracks if "wind_max_ms" in t]
    mslp_values = [float(t["mslp_min_hpa"]) for t in tracks if "mslp_min_hpa" in t]
    classes: dict[str, int] = {}
    for t in tracks:
        key = str(t.get("classification") or "unknown")
        classes[key] = classes.get(key, 0) + 1
    summary = {
        "track_count": len(tracks),
        "classification_counts": classes,
        "wind_max_ms": max(wind_values) if wind_values else None,
        "wind_mean_ms": sum(wind_values) / len(wind_values) if wind_values else None,
        "mslp_min_hpa": min(mslp_values) if mslp_values else None,
        "mslp_mean_hpa": sum(mslp_values) / len(mslp_values) if mslp_values else None,
        "tracks": tracks,
    }
    json_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    lines = [
        "# Atlantic tctracker summary",
        "",
        f"- Track count: {summary['track_count']}",
        f"- Classification counts: {summary['classification_counts']}",
        f"- Max wind: {summary['wind_max_ms']:.2f} m/s" if summary["wind_max_ms"] is not None else "- Max wind: n/a",
        f"- Min MSLP: {summary['mslp_min_hpa']:.0f} hPa" if summary["mslp_min_hpa"] is not None else "- Min MSLP: n/a",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def dry_run_payload(config: TCTrackerConfig) -> dict[str, Any]:
    return {
        "config": {**asdict(config), "output_dir": str(config.output_dir), "dates": list(config.dates), "members": list(config.members)},
        "target_count": len(config.targets),
        "targets": [
            {"tag": t.tag, "date": t.date, "member": t.member, "tar": str(t.tar_path), "command": build_tctracker_command(config, t)}
            for t in config.targets
        ],
    }


def render_slurm_script(config: TCTrackerConfig, code_root: str, venv_activate: str, *, qos: str = "nf", time: str = "04:00:00", mem: str = "16G", cpus: int = 4, python_executable: str = "/home/ecm5702/dev/.ds-260612/bin/python") -> str:
    cmd = [
        python_executable, "-m", "eval.cli", "tctracker",
        "--lane", config.lane,
        "--host", "atos_ac",
        "--expver", config.expver,
        "--output-dir", str(config.output_dir),
        "--dates", ",".join(config.dates),
        "--members", ",".join(str(m) for m in config.members),
        "--time", config.time,
        "--start-step", str(config.start_step),
        "--end-step", str(config.end_step),
        "--step-interval", str(config.step_interval),
        "--grid", str(config.grid),
        "--class", config.fdb_class,
        "--type", config.fdb_type,
        "--stream", config.stream,
        "--vorticity", str(config.vorticity).lower(),
    ]
    if config.overwrite:
        cmd.append("--overwrite")
    return "\n".join([
        "#!/bin/bash",
        f"#SBATCH --job-name=tctracker-{config.expver}",
        f"#SBATCH --qos={qos}",
        f"#SBATCH --time={time}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --output={config.logs_dir}/tctracker_%j.out",
        "set -euo pipefail",
        f"module load {shlex.quote(config.module)}",
        f"export PYTHONPATH={shlex.quote(code_root)}:${{PYTHONPATH:-}}",
        f"cd {shlex.quote(code_root)}",
        " ".join(shlex.quote(part) for part in cmd),
        "",
    ])
