"""Rows, result discovery, table writers and the "Not available" convention."""
from __future__ import annotations

import csv
import json
import math
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

NA = "Not available"


def na(reason: str) -> str:
    """The text of a missing cell or section."""
    return f"{NA}: {reason}"


# --------------------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------------------

def fnum(v: Any) -> float:
    """float(v), NaN for '', None or junk."""
    if v is None:
        return math.nan
    if isinstance(v, (int, float)):
        return float(v)
    try:
        s = str(v).strip()
        return float(s) if s else math.nan
    except ValueError:
        return math.nan


def finite(v: Any) -> bool:
    return math.isfinite(fnum(v))


def fmt(v: Any, nd: int = 3, *, signed: bool = False, scale: float = 1.0) -> str:
    """A number for a table; 'n/a' for a missing one. Strings pass through."""
    if isinstance(v, str) and not _is_number_text(v):
        return v
    if isinstance(v, (int, np_int_types())) and not isinstance(v, bool) and scale == 1.0:
        return f"{int(v):+d}" if signed else str(int(v))
    x = fnum(v)
    if not math.isfinite(x):
        if x in (math.inf, -math.inf):
            return "inf" if x > 0 else "-inf"
        return "n/a"
    x *= scale
    if abs(x) < 1e-12:
        x = 0.0
    if x != 0 and (abs(x) >= 10 ** 6 or abs(x) < 10 ** (-nd)):
        s = f"{x:.{max(nd - 1, 1)}e}"
        return ("+" + s) if signed and x > 0 else s
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def np_int_types() -> tuple:
    try:
        import numpy as np
        return (np.integer,)
    except ImportError:  # pragma: no cover
        return ()


def _is_number_text(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def pct(row: float, ref: float) -> float:
    """100 * (row - ref) / ref."""
    row, ref = fnum(row), fnum(ref)
    if not (math.isfinite(row) and math.isfinite(ref)) or ref == 0:
        return math.nan
    return 100.0 * (row - ref) / ref


# --------------------------------------------------------------------------------------
# rows file
# --------------------------------------------------------------------------------------

@dataclass
class Row:
    label: str
    arm: str = ""
    sampler: str = ""
    calls: int | None = None
    sets: list[str] = field(default_factory=list)
    step: int | None = None
    pred_dirs: dict[str, str] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    @property
    def sampler_dir(self) -> str:
        return sanitize_spec(self.sampler)


def sanitize_spec(spec: str) -> str:
    """Directory name of a sampler spec, as predict_members.py makes it."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(spec or ""))


_STEP_RE = re.compile(r"step_?0*(\d+)")


def _as_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    if isinstance(v, int):
        return v
    m = _STEP_RE.search(str(v))
    if m:
        return int(m.group(1))
    try:
        return int(float(str(v).replace("k", "000")))
    except ValueError:
        return None


def parse_row(item: dict) -> Row:
    item = dict(item)
    label = str(item.pop("label", item.pop("name", "")))
    arm = str(item.pop("arm", "") or "")
    sampler = str(item.pop("sampler", item.pop("spec", "")) or "")
    calls = _as_int(item.pop("calls", None))
    step = None
    for key in ("step", "checkpoint_step", "ckpt_step"):
        if key in item:
            step = _as_int(item.pop(key))
            break
    if step is None and "/" in arm:  # "edm/step_150000"
        arm, _, rest = arm.partition("/")
        step = _as_int(rest)
    sets_raw = item.pop("sets", [])
    sets: list[str] = []
    pred_dirs: dict[str, str] = {}
    if isinstance(sets_raw, str):
        sets = [s.strip() for s in sets_raw.replace(";", ",").split(",") if s.strip()]
    elif isinstance(sets_raw, dict):
        for k, v in sets_raw.items():
            sets.append(str(k))
            if isinstance(v, str) and v:
                pred_dirs[str(k)] = v
            elif isinstance(v, dict):
                for kk in ("predictions", "predictions_dir", "pred_dir", "path"):
                    if v.get(kk):
                        pred_dirs[str(k)] = str(v[kk])
                        break
    elif isinstance(sets_raw, (list, tuple)):
        sets = [str(s) for s in sets_raw]
    for key in ("predictions", "pred_dirs", "predictions_dirs"):
        v = item.pop(key, None)
        if isinstance(v, dict):
            pred_dirs.update({str(k): str(p) for k, p in v.items() if p})
    return Row(label=label, arm=arm, sampler=sampler, calls=calls, sets=sets, step=step,
               pred_dirs=pred_dirs, extra=item)


def load_rows(path: str | Path) -> list[Row]:
    """Rows from YAML (or JSON): a list of rows, {"rows": [...]}, or {label: {...}}."""
    text = Path(path).read_text()
    try:
        import yaml
        data = yaml.safe_load(text)
    except ImportError:  # pragma: no cover
        data = json.loads(text)
    if isinstance(data, dict):
        if isinstance(data.get("rows"), list):
            items = data["rows"]
        elif isinstance(data.get("rows"), dict):
            items = [dict(v or {}, label=k) for k, v in data["rows"].items()]
        else:
            items = [dict(v or {}, label=k) for k, v in data.items() if isinstance(v, dict)]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"{path}: rows file is neither a list nor a mapping")
    rows = [parse_row(it) for it in items if isinstance(it, dict)]
    rows = [r for r in rows if r.label]
    if not rows:
        raise ValueError(f"{path}: no rows with a label")
    return rows


# --------------------------------------------------------------------------------------
# result discovery
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Found:
    set: str
    view: str          # "" for the default <row>/evaluators/<name>
    path: Path

    @property
    def key(self) -> str:
        return f"{self.set}/{self.view}" if self.view else self.set


def read_json(path: Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


class Context:
    """Everything a section needs: the rows, the reference, where the results live."""

    def __init__(self, *, eval_root: Path, rows: list[Row], reference: str, out_dir: Path,
                 pred_roots: dict[str, Path] | None = None, options: dict | None = None):
        self.eval_root = Path(eval_root)
        self.rows = rows
        self.by_label = {r.label: r for r in rows}
        if reference not in self.by_label:
            raise ValueError(f"reference {reference!r} is not a row label "
                             f"({', '.join(self.by_label)})")
        self.reference = reference
        self.out_dir = Path(out_dir)
        self.csv_dir = self.out_dir / "csv"
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.pred_roots = dict(pred_roots or {})
        self.options = dict(options or {})
        self.cache: dict = {}

    @property
    def ref_row(self) -> Row:
        return self.by_label[self.reference]

    def opt(self, key: str, default: Any = None) -> Any:
        v = self.options.get(key)
        return default if v is None else v

    # ----- evaluator outputs -----
    def set_dirs(self, label: str) -> list[str]:
        """Sets in which this row has an output directory (the row's own sets first)."""
        if not self.eval_root.is_dir():
            return []
        found = sorted(p.name for p in self.eval_root.iterdir()
                       if p.is_dir() and (p / label).is_dir())
        own = [s for s in self.by_label[label].sets if s in found] if label in self.by_label else []
        return own + [s for s in found if s not in own]

    def results(self, label: str, evaluator: str, required: str | Iterable[str]) -> list[Found]:
        """Every results directory of ``evaluator`` for this row holding ``required``.

        Accepted, under ``<EVAL_ROOT>/<set>/<label>/``: ``evaluators/<name>`` (view ""),
        ``evaluators/<name>__<view>`` or ``evaluators/<name>-<view>``, and
        ``<view>/evaluators/<name>`` (one extra level, for a second box).
        """
        req = [required] if isinstance(required, str) else list(required)
        out: list[Found] = []
        for s in self.set_dirs(label):
            base = self.eval_root / s / label
            cands: list[tuple[str, Path]] = []
            ev = base / "evaluators"
            if ev.is_dir():
                for p in sorted(ev.iterdir()):
                    if not p.is_dir():
                        continue
                    if p.name == evaluator:
                        cands.append(("", p))
                    else:
                        for sep in ("__", "-"):
                            if p.name.startswith(evaluator + sep):
                                cands.append((p.name[len(evaluator) + len(sep):], p))
            for sub in sorted(base.iterdir()) if base.is_dir() else []:
                p = sub / "evaluators" / evaluator
                if sub.is_dir() and sub.name != "evaluators" and p.is_dir():
                    cands.append((sub.name, p))
            for view, p in cands:
                if any((p / r).exists() for r in req):
                    out.append(Found(s, view, p))
        return out

    def results_by_key(self, label: str, evaluator: str, required) -> dict[str, Found]:
        return {f.key: f for f in self.results(label, evaluator, required)}

    def where(self, label: str, evaluator: str) -> str:
        return f"{self.eval_root}/<set>/{label}/evaluators/{evaluator}"

    # ----- prediction files -----
    def pred_dir(self, row: Row, set_name: str) -> Path | None:
        """The predictions directory of a row in a set, or None."""
        if set_name in row.pred_dirs:
            p = Path(row.pred_dirs[set_name])
            return p if p.is_dir() else None
        for sd in self._step_dirs(row, set_name):
            for samp in (row.sampler_dir, row.sampler):
                p = sd / samp / "predictions"
                if samp and p.is_dir() and any(p.glob("predictions_*.nc")):
                    return p
        return None

    def _step_dirs(self, row: Row, set_name: str) -> list[Path]:
        """Candidate ``<root>/<arm>/step_NNNNNN`` directories of a row in a set."""
        roots: list[Path] = []
        if set_name in self.pred_roots:
            roots.append(self.pred_roots[set_name])
        if "" in self.pred_roots:
            base = self.pred_roots[""]
            roots += ([base / set_name] if set_name else []) + [base]
        out = []
        for root in roots:
            arm_dir = root / row.arm
            if not row.arm or not arm_dir.is_dir():
                continue
            steps = ([arm_dir / f"step_{row.step:06d}"] if row.step is not None
                     else sorted(arm_dir.glob("step_*")))
            out += [sd for sd in steps if sd.is_dir() and (
                (sd / row.sampler_dir).is_dir() or (sd / row.sampler).is_dir() or row.step is not None)]
        return out

    def costs_file(self, row: Row, set_name: str) -> Path | None:
        """costs.jsonl next to the step directory of a row (``<arm>/step_NNNNNN/costs.jsonl``)."""
        if set_name in row.pred_dirs:
            c = Path(row.pred_dirs[set_name]).parent.parent / "costs.jsonl"
            if c.exists():
                return c
        for sd in self._step_dirs(row, set_name):
            if (sd / "costs.jsonl").exists():
                return sd / "costs.jsonl"
        return None


# --------------------------------------------------------------------------------------
# tables and sections
# --------------------------------------------------------------------------------------

def md_table(header: list[str], rows: list[list[Any]]) -> str:
    def cell(v):
        s = v if isinstance(v, str) else fmt(v)
        return s.replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        cells = [cell(v) for v in r]
        # a row whose remaining cells all carry one "Not available" text shows it once
        for i, c in enumerate(cells):
            if c.startswith(NA) and len(cells) - i > 1 and all(x == c for x in cells[i:-1]) \
                    and (cells[-1] == c or not cells[-1].startswith(NA)):
                cells = cells[:i + 1] + [""] * (len(cells) - i - 2) + (
                    [""] if cells[-1] == c else [cells[-1]])
                break
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if (isinstance(v, float) and math.isnan(v)) else
                        (f"{v:.6g}" if isinstance(v, float) else v) for v in r])
    return path


def table(ctx: Context, name: str, header: list[str], rows: list[list[Any]]) -> str:
    """Write ``csv/<name>.csv`` and return the markdown table (with a pointer to the CSV)."""
    write_csv(ctx.csv_dir / f"{name}.csv", header, rows)
    if not rows:
        return na("no rows") + "\n"
    return md_table(header, rows) + f"\nCSV: `csv/{name}.csv`\n"


@dataclass
class SectionResult:
    title: str
    md: str
    summary: dict[str, list[str]] = field(default_factory=dict)   # label -> fragments

    def add(self, label: str, text: str) -> None:
        self.summary.setdefault(label, []).append(text)


def run_section(title: str, fn: Callable[..., SectionResult], *args, **kwargs) -> SectionResult:
    """Run one part; an exception becomes a "Not available" section, never a crash."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc(limit=4)
        return SectionResult(title, f"# {title}\n\n{na(f'{type(exc).__name__}: {exc}')}\n\n"
                                    f"```\n{tb}```\n")


def guarded(md_parts: list[str], heading: str, fn: Callable[[], str]) -> None:
    """Append one sub-section; an exception becomes its "Not available" text."""
    try:
        md_parts.append(fn())
    except Exception as exc:  # noqa: BLE001
        md_parts.append(f"{heading}\n\n{na(f'{type(exc).__name__}: {exc}')}\n")


class row_guard:
    """``with row_guard(rows, prefix, ncols, res, label, part):`` -- an exception inside the
    block becomes one "Not available" table row (and summary fragment) and is swallowed, so
    one unreadable row never blanks the other rows of a table."""

    def __init__(self, rows: list, prefix: list, ncols: int, res: "SectionResult | None" = None,
                 label: str | None = None, part: str = ""):
        self.rows, self.prefix, self.ncols = rows, list(prefix), ncols
        self.res, self.label, self.part = res, label, part

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            return False
        if not issubclass(et, Exception):
            return False
        msg = na(f"unreadable input: {et.__name__}: {ev}")
        self.rows.append(self.prefix + [msg] * self.ncols)
        if self.res is not None and self.label:
            self.res.add(self.label, f"{self.part}: {msg}" if self.part else msg)
        return True


def se_of(sd: Any, n: Any) -> float:
    sd, n = fnum(sd), fnum(n)
    if not (math.isfinite(sd) and math.isfinite(n)) or n < 2:
        return math.nan
    return sd / math.sqrt(n)


def mean_se(values: Iterable[float]) -> tuple[float, float, int]:
    v = [float(x) for x in values if math.isfinite(fnum(x))]
    n = len(v)
    if n == 0:
        return math.nan, math.nan, 0
    m = sum(v) / n
    if n < 2:
        return m, math.nan, n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    return m, sd / math.sqrt(n), n


VAR_ALIASES = {"10ff": "ws", "ws10": "ws", "wind10m": "ws", "10si": "ws"}


def canon_var(name: str) -> str:
    return VAR_ALIASES.get(str(name), str(name))
