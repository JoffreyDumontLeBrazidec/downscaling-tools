# Lanes

A **lane** is one evaluation pipeline configuration: which checkpoint family, which
dates/steps/members, which evaluators, which host, and — for prepml lanes — which MARS
streams. `eval.cli` takes a lane by *name*:

```bash
python -m eval.cli run --lane o320_o1280 --checkpoint /path/to.ckpt --host atos_ac
```

The name maps to `eval/config/lanes/<name>.yaml`. Nothing globs this directory; a lane
exists only if something names it.

## The canonical roots

| Lane | Grid step |
|---|---|
| `o48_o96` | ~200 km → ~100 km |
| `o96_o320` | ~100 km → ~30 km |
| `o320_o1280` | ~30 km → ~9 km |
| `o1280_o2560` | ~9 km → ~4.5 km |

These four are **independent problems, never a cascade** — you do not chain o96→o320 into
o320→o1280. `tc_o320_o1280` is a fifth root: a fast regional TC-extremes harness, not a
public scoreboard lane.

## `base:` inheritance

`eval/config/loader.py::load_lane` resolves `base:` recursively and deep-merges the child
over the parent, so a variant only states what differs:

```yaml
# o48_o96_debug.yaml -- the whole file
base: o48_o96

spectra_ecmwf:
  steps: [24]
  members: [1]
```

Depth is unlimited and a base may itself have a base.

**The consequence that matters: a lane nobody runs may still be somebody's parent.**
"Nothing references it" is not sufficient grounds to delete a lane. See
[Retiring a lane](../../README.md#retiring-a-lane).

## Generated lanes: `_ladder_*`, `_evoref_*`

These are **not** hand-written and are **not** tracked in git. `eval/jobs/ladder.py::derive_lane`
materialises them from the ladder profiles in `eval/config/ladder/*.yaml`, because
`eval.cli --lane` takes a name, so a profile's pinned evaluator knobs (`spectra`,
`spectra_ecmwf`, `tc`, `probabilistic` — `LANE_OVERRIDE_KEYS`) have no other route into the
evaluators.

Never hand-edit one; the next `ladder score` overwrites it. Edit the profile instead. If a
`_ladder_*` file is missing, run `ladder score` and it reappears.

## The canonical anemoi-inference reference

`eval/config/anemoi_inference_reference/O{48,96,320,1280}.yaml` are the single source of truth
for `prepml.input` / `prepml.output` MARS identity (class / stream / type / grid), keyed by
input grid because each downscaling input grid maps to exactly one task family.

`load_lane` **overwrites** the lane's values with the reference and warns on any disagreement.
If a lane disagrees, fix the lane — do not edit the reference to match. This guard exists
because of the eecdb127 eefo/enfo stream-drift incident (2026-07-01), where a lane silently
carried the wrong input stream through a full campaign.

A `WARNING ... no canonical anemoi-inference reference` on stderr means the guard is **not**
armed for that grid and the lane's own values are being trusted. Treat it as a stop.

## Naming

```
<input-grid>_<output-grid>[_<checkpoint-or-campaign>][_<arm>]
```

- `tc_` prefix — regional TC-box harness lanes
- `_ladder_`, `_evoref_` — reserved for generated files (gitignored)

## Retiring a lane

See the [retirement convention in `eval/README.md`](../../README.md#retiring-a-lane). The
short version: a campaign's arms are deleted once the campaign is scored — the scoreboard row
plus git history is the record, and `git rm` keeps both.
