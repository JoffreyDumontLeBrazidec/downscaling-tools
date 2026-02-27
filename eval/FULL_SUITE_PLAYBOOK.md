# Full Eval Suite Playbook

## One Command
```bash
/home/ecm5702/dev/downscaling-tools/eval/jobs/launch_full_eval_suite.sh --expver <EXPVER>
```

Example:
```bash
/home/ecm5702/dev/downscaling-tools/eval/jobs/launch_full_eval_suite.sh --expver j24v
```

This submits:
- `eval.run mars-expver` (predictions + region plots)
- quaver
- spectra compute
- spectra comparison plots (`expver + eefo_o96 + enfo_o320`)
- TC retrieval (missing only)
- TC plotting (all 5 cyclones): normalized PDF + member field maps

All plots are written in:
`/home/ecm5702/perm/eval/<EXPVER>/`
TC member maps are in:
`/home/ecm5702/perm/eval/<EXPVER>/tc_members/`

## Classic Codex Command (Background + Auto-Retry)
Use this if you want one command that runs in background and auto-retries failed jobs:

```bash
/home/ecm5702/dev/downscaling-tools/eval/jobs/codex_eval --expver <EXPVER>
```

Outputs/state:
- background log: `/home/ecm5702/perm/eval/<EXPVER>/logs/autopilot_background.log`
- live state: `/home/ecm5702/perm/eval/<EXPVER>/logs/autopilot_state.json`

## Recommended Prompt For Next Codex Session
Use this exact instruction:

```text
Run the full eval suite for expver <EXPVER> in background using:
/home/ecm5702/dev/downscaling-tools/eval/jobs/codex_eval --expver <EXPVER>

Then monitor all jobs to completion, fix failures if needed, and confirm final artifacts in /home/ecm5702/perm/eval/<EXPVER>/.
```

## Useful Monitoring
The launcher prints all job IDs. Then:
```bash
squeue -j <comma-separated-jobids>
sacct -j <comma-separated-jobids> --format=JobID,JobName%24,State,Elapsed,ExitCode
```
