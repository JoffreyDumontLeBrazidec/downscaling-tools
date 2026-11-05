"""o2560 box read: compare the AG crop chain's config (A) with the Jupiter full-box chain's config (B), key by key.
The read runs both rows on B's config (configs/db_o2560_read.yaml = a6eb3db6), so every key that shapes the model, its
inputs and outputs, the graph or the data split must be equal. Allowed to differ: asset paths (Atos vs Jupiter copies, sha
equal per the crop sandbox's asset_sha check), crop / compile / hardware / run length / naming keys (training-only).
Prints CONFIG_COMPARE_PASS or CONFIG_COMPARE_FAIL and writes the full list of differing keys.
Usage: python compare_configs.py <A.yaml> <B.yaml> <out.txt>"""
import sys
import yaml


def flat(d, p=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{p}.{k}" if p else str(k)))
    else:
        out[p] = d
    return out


a, b = (flat(yaml.safe_load(open(f))) for f in sys.argv[1:3])
ALLOWED = ("system.input.", "system.output.", "system.hardware.", "training.crop", "model.compile", "training.recompile_limit",
           "training.max_steps", "training.lr.iterations", "dataloader.limit_batches", "diagnostics.log.", "training.run_id",
           "training.fork_run_id", "dataloader.num_workers", "dataloader.prefetch_factor", "dataloader.batch_size",
           "diagnostics.checkpoint", "training.revive_dead_units", "dataloader.read_group_size", "dataloader.pin_memory")
MUST = ("model.", "graph.", "data.", "dataloader.training", "dataloader.validation", "dataloader.test", "dataloader.steps",
        "dataloader.training_date", "dataloader.valid_date", "dataloader.test_date", "training.ema", "training.model_task")
diff = sorted(k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>"))
bad = [k for k in diff if k.startswith(MUST) and not k.startswith(ALLOWED)]
lines = [f"{k}: A={a.get(k, '<absent>')!r} B={b.get(k, '<absent>')!r}" + ("   <-- NOT ALLOWED" if k in bad else "") for k in diff]
lines.append(f"{len(diff)} keys differ, {len(bad)} in model/graph/data/split keys outside the allowed list")
lines.append("CONFIG_COMPARE_PASS" if not bad else "CONFIG_COMPARE_FAIL")
open(sys.argv[3], "w").write("\n".join(lines) + "\n")
print("\n".join(lines[-2:]))
sys.exit(0 if not bad else 1)
