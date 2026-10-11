"""Fake training for the chain_next tests: writes <root>/<run>/last.ckpt at start+inc (no torch)."""
import io, os, pickle, sys, zipfile
root, run_id, start, inc = sys.argv[1], sys.argv[2] or "run1", int(sys.argv[3]), int(sys.argv[4])
if os.environ.get("TOY_FAIL"):
    sys.exit("toy: failing before the first checkpoint")
os.makedirs(os.path.join(root, run_id), exist_ok=True)
buf = io.BytesIO()
pickle.dump({"global_step": start + inc}, buf, protocol=2)
with zipfile.ZipFile(os.path.join(root, run_id, "last.ckpt"), "w") as zf:
    zf.writestr("archive/data.pkl", buf.getvalue())
