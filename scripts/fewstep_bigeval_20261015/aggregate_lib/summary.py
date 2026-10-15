"""The one-page summary: per row, grouped by network calls, one line per protocol part."""
from __future__ import annotations

from .common import Context, SectionResult, na

PARTS = [("1. Structure", "Structure"), ("2. Spectra", "Spectra"), ("3. Extremes", "Extremes"),
         ("4. Scores", "Scores"), ("5. Costs", "Costs")]
CALL_GROUPS = [1, 2, 4, 59]


def build(ctx: Context, sections: dict[str, SectionResult]) -> str:
    ref = ctx.reference
    lines = ["# Few-step big eval, o96->o320: summary\n",
             f"Reference: `{ref}`. Rules as pre-registered (note 20261015 section 5); numbers are row "
             "vs reference unless said. Scores tags: [W] established worse > 1 %, [B] established "
             "better > 1 %, [=] established within 1 % (equal in practice), [~] tie; 'k/n' = dates on "
             "which the reference is better. Spectra 'pass' = no 10u/10v/2t bin off the reference, "
             "full field or residual. Maps are read by eye and are not in this summary.\n"]
    groups: dict = {}
    for r in ctx.rows:
        groups.setdefault(r.calls if r.calls in CALL_GROUPS else "other", []).append(r)
    order = [g for g in CALL_GROUPS if g in groups] + (["other"] if "other" in groups else [])
    for g in order:
        lines.append(f"## {g} call{'s' if g != 1 else ''} per member\n" if g != "other"
                     else "## Other call counts\n")
        for r in groups[g]:
            tag = " (reference)" if r.label == ref else ""
            lines.append(f"### `{r.label}`{tag}: {r.arm} {r.sampler}, {r.calls} calls\n")
            for key, name in PARTS:
                sec = sections.get(key)
                if sec is None:
                    text = na("section not run")
                else:
                    frags = sec.summary.get(r.label)
                    text = " / ".join(frags) if frags else na("no output for this row")
                lines.append(f"- **{name}**: {text}")
            lines.append("")
    return "\n".join(lines) + "\n"
