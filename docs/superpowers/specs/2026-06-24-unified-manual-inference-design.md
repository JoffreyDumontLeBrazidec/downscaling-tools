# Unified Manual Inference Repair Design

Status: approved design, pending implementation plan review.

## Goal

Make the b785bf12 O320 to O1280 manual bundle path use the same four-GPU unified runner topology as the successful PrepML path, while preserving local NetCDF output with y, x_interp, and y_pred.

## Evidence and root cause

The failed manual jobs 30864645 and 30864649 invoked eval.predict.main, which loaded the serialized inference companion and called predict_step directly. It passed a communication group but skipped UnifiedDownscalingRunner construction and its ParallelRunnerMixin setup.

The direct path reached 38.81 GiB on an A100-40GB and failed on a further 3.15 GiB allocation in both BF16 and FP32. The documented unified runner uses the same checkpoint and four GPUs, initializes the model group, and completed the 30-step BF16 sampler at 18.41 GiB peak.

## Options considered

1. Patch the direct companion loader to reconstruct all model-parallel state. This duplicates the site runner and is high risk for parity.
2. Use the established UnifiedDownscalingRunner inside manual inference. This is the recommended path because it preserves the proven model loading and parallel initialization.
3. Treat PrepML output as manual output. Rejected because it eliminates the required independent parity path.

## Chosen design

Add an explicit unified backend to eval.predict.main. It accepts a generated anemoi runner configuration, imports the registered downscaling_unified runner, creates the runner, and supplies its initialized model interface to the existing bundle loop and NetCDF writer.

The b785 launcher will generate the anemoi configuration from its run-scoped PrepML configuration without submitting PrepML, then invoke the unified manual backend with four Slurm ranks. The legacy backend remains available and unchanged for checkpoint families that do not request the unified backend.

The unified backend will expose the interface metadata needed to map bundle channels and will export canonical y_pred, y, and x_interp. It will fail early when the runner config or registered runner is unavailable.

## Validation

Unit tests will cover unified backend selection, runner initialization inputs, and explicit failure for a missing runner configuration. A four-GPU one-date one-lead two-member BF16 smoke and a separate FP32 smoke must each write one finite NetCDF with the ten requested states and canonical y_pred, y, and x_interp.

No full campaign, evaluator chain, FDB write, or scoreboard update is in scope until both manual smokes pass and match the accepted output contract.

## Baseline note

The isolated main baseline passes the runtime preflight but has seven pre-existing failures in manual_inference/tests/test_prediction.py. Their fixtures target APIs absent from commit 3fc12da, so the repair test will be added independently and the unrelated failures will be reported rather than modified.
