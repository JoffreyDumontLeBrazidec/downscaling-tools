"""Fake running segment for the handover tests: rewrites <ckpt> every <period> s with step += 100 (no torch)."""
import io, os, pickle, sys, time, zipfile
ckpt, step, period = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
while True:
    time.sleep(period)
    step += 100
    buf = io.BytesIO()
    pickle.dump({"global_step": step}, buf, protocol=2)
    tmp = ckpt + ".writing"
    with zipfile.ZipFile(tmp, "w") as zf:
        zf.writestr("archive/data.pkl", buf.getvalue())
    os.replace(tmp, ckpt)
