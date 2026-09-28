# Quarantine, 2026-09-28

Code retired from the evaluation framework on 2026-09-28. It is kept, not
deleted, so its history and logic stay readable next to the code that replaced
it. None of it is importable: the directories are no longer under
`eval/evaluators/`, and `20260928` is not a valid Python package name. Its tests
are not collected (`norecursedirs` in `pytest.ini`).

The CLI treats each retired evaluator as a tombstone: `eval.cli ... --only <name>`
prints the replacement and exits with status 1, and a retired name left in a lane
YAML group is skipped with a warning. The list, with replacements, is
`eval/evaluators/registry.py`.

| Directory | What it was | Use instead |
|---|---|---|
| `spectra/` | HEALPix proxy power spectra, and the source of the old `spectra_*` scoreboard rows | `spectra_ecmwf_v2` (rows `spectra_v2_*`) |
| `spectra_ecmwf/` | ECMWF spectral transform on pole-masked templates (version one) | `spectra_ecmwf_v2` |
| `sigma/` | Older per-noise-level loss sweep | `sigma_loss` |
| `obs_crps/` | Fair CRPS against surface stations, a cheap stand-in for quaver | `quaver` |
| `mechanistic/` | Stub that only created empty directories | nothing |
| `intermediate/` | Plots of intermediate diffusion steps | nothing |
| `interp/` | Renderer for interpretability PDFs computed outside the CLI | nothing (`interp.viz` still exists) |
| `leadtime/` | Per-lead-time scores, never registered and never run | nothing |
| `_backends/leadtime/` | The leadtime evaluator's compute backend | nothing |
| `_backends/migrate_reference_windows.py` | One-off migration of version-one spectra reference caches; it imported a function the version-one runner no longer had | nothing |

Backends that retired evaluators used but other code still needs stay in place:
`eval/_backends/sigma_evaluator`, `eval/_backends/obs_crps` (used by
`tools/station_head`), `eval/_backends/plot_intermediate`,
`eval/_backends/weight_diagnostics` and `eval/_backends/spectra`.
