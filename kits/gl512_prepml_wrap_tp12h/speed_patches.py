"""Speed-test patches on the prepml downscaling_unified runner (test-only).

Env knobs (all optional):
  SPEED_TF32=1                 torch.backends.cuda.matmul.allow_tf32 = True (and cudnn)
  SPEED_FP32_MODULES=a+b       dotted submodule names (relative to interface.model) run with autocast
                               disabled on float32 inputs; output left float32
  SPEED_PROFILE=1              cuda-synchronised wall time of encoder / processor / decoder per call
  SPEED_DUMP_NPY=<dir>         save the raw out_hres tensor of each draw (rank 0) as .npy
  SPEED_LNROW=1                ConditionalLayerNorm scale/bias from one conditioning row (see _patch_cond_layernorm)
  SPEED_EDGEPAD=8              zero-pad lin_edge inputs to 8 columns so the edge GEMM uses tensor cores (see _patch_edge_pad)
  SPEED_MANIFEST=<file>        after the first draw, rank 0 writes the list of imported / mapped files (for warm_manifest.sh)
  SPEED_PROBE_CALLS=1,20,40    on-path single-call precision probe at these denoiser calls of lead SPEED_PROBE_LEAD (24)
  SPEED_PROBE_VARIANTS="tag=amp[:guard+guard];..."   amp in fp32|tf32|bf16|fp16 (see _install_probe)
Autocast itself is the runner's own knob CODEX_UNIFIED_FORWARD_DTYPE (bfloat16|float16).
Every draw logs: seconds, denoiser calls, s/call, peak GPU memory (allocated / reserved) on this rank.
"""
from __future__ import annotations

import logging
import os
import time

import torch

LOG = logging.getLogger("speed_patches")


def _get(root, dotted):
    obj = root
    for part in dotted.split("."):
        obj = obj[part] if isinstance(obj, (torch.nn.ModuleDict, dict)) else getattr(obj, part)
    return obj


def _set(root, dotted, new):
    parts = dotted.split(".")
    parent = _get(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    if isinstance(parent, torch.nn.ModuleDict):
        parent[parts[-1]] = new
    else:
        setattr(parent, parts[-1], new)


def _to_fp32(o):
    if torch.is_tensor(o):
        return o.float() if o.is_floating_point() else o
    if isinstance(o, tuple):
        return tuple(_to_fp32(v) for v in o)
    if isinstance(o, list):
        return [_to_fp32(v) for v in o]
    if isinstance(o, dict):
        return {k: _to_fp32(v) for k, v in o.items()}
    return o


class FP32(torch.nn.Module):
    """Run the wrapped module in float32 with autocast off while its name is in FP32.ACTIVE (inputs cast up, nested tuples
    and dicts included; output left float32). Inactive = plain pass-through."""

    ACTIVE: set = set()

    def __init__(self, inner, name):
        super().__init__()
        self.inner = inner
        self.guard_name = name

    def forward(self, *args, **kwargs):
        if self.guard_name not in FP32.ACTIVE:
            return self.inner(*args, **kwargs)
        with torch.autocast(device_type="cuda", enabled=False):
            return self.inner(*_to_fp32(args), **_to_fp32(kwargs))

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)


class AMP16(torch.nn.Module):
    """Run the wrapped module under bfloat16 autocast while its name is in AMP16.ACTIVE, even inside a float32 guard;
    the output is cast back to float32. Inactive = plain pass-through."""

    ACTIVE: set = set()

    def __init__(self, inner, name):
        super().__init__()
        self.inner = inner
        self.guard_name = name

    def forward(self, *args, **kwargs):
        if self.guard_name not in AMP16.ACTIVE:
            return self.inner(*args, **kwargs)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            out = self.inner(*args, **kwargs)
        return _to_fp32(out)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)


def _parse_variants():
    """SPEED_PROBE_VARIANTS="tag=amp[:guard+guard[|amp16mod+amp16mod]];..."  amp in fp32|tf32|bf16|fp16|tf32bf16
    (tf32bf16 = bf16 autocast with TF32 on for whatever runs float32). guards run float32; amp16 modules run bf16 even
    inside a guard. Static equivalents: SPEED_FP32_MODULES and SPEED_AMP16_MODULES."""
    out = []
    for item in os.environ.get("SPEED_PROBE_VARIANTS", "").split(";"):
        item = item.strip()
        if not item:
            continue
        tag, spec = item.split("=", 1)
        amp, _, rest = spec.partition(":")
        guards, _, amps = rest.partition("|")
        out.append((tag, amp, [g for g in guards.split("+") if g], [g for g in amps.split("+") if g]))
    return out


def _prepare(model):
    inner = getattr(model, "model", model)
    if getattr(inner, "_speed_prepared", False):
        return inner
    names = [n.strip() for n in os.environ.get("SPEED_FP32_MODULES", "").replace("+", ",").split(",") if n.strip()]
    variants = _parse_variants()
    amp_names = [n.strip() for n in os.environ.get("SPEED_AMP16_MODULES", "").replace("+", ",").split(",") if n.strip()]
    amp_wanted = list(dict.fromkeys(amp_names + [g for _, _, _, am in variants for g in am]))
    for name in sorted(amp_wanted, key=lambda n: -n.count(".")):
        mod = _get(inner, name)
        _set(inner, name, AMP16(mod, name))
        LOG.warning("SPEED bf16 island installed on %s (%s), active=%s", name, type(mod).__name__, name in amp_names)
    AMP16.ACTIVE = set(amp_names)
    inner._speed_static_amps = set(amp_names)
    wanted = list(dict.fromkeys(names + [g for _, _, gs, _ in variants for g in gs]))
    # wrap inner-most names first so that a parent wrapper never hides a child path
    for name in sorted(wanted, key=lambda n: -n.count(".")):
        mod = _get(inner, name)
        _set(inner, name, FP32(mod, name))
        LOG.warning("SPEED fp32 guard installed on %s (%s), active=%s", name, type(mod).__name__, name in names)
    FP32.ACTIVE = set(names)
    inner._speed_static_guards = set(names)
    if variants and os.environ.get("SPEED_PROBE_CALLS", ""):
        _install_probe(inner, variants)
    if os.environ.get("SPEED_TORCHPROF", ""):
        _install_torchprof(inner)
    if os.environ.get("SPEED_PROFILE", "") == "1":
        acc = {}

        def _timed(tag, mod):
            fwd = mod.forward

            def _f(*a, **k):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = fwd(*a, **k)
                torch.cuda.synchronize()
                acc[tag] = acc.get(tag, 0.0) + time.perf_counter() - t0
                return out

            mod.forward = _f

        for tag, name in (("encoder", "encoder.out_hres"), ("processor", "processor"), ("decoder", "decoder.out_hres")):
            try:
                _timed(tag, _get(inner, name))
            except Exception as e:  # noqa: BLE001
                LOG.warning("SPEED profile: cannot hook %s: %s", name, e)
        inner._speed_acc = acc
    if os.environ.get("SPEED_LNROW", "") == "1":
        _patch_cond_layernorm()
    if os.environ.get("SPEED_EDGEPAD", ""):
        _patch_edge_pad(int(os.environ["SPEED_EDGEPAD"]))
    inner._speed_prepared = True
    return inner


def _write_manifest(path):
    """List every file this process has imported or mapped (module sources, their .pyc, shared libraries) that lives on a
    network file system, for warm_manifest.sh to pre-read in parallel at the start of later tasks."""
    import importlib.util
    import sys

    files = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        files.add(f)
        if f.endswith(".py"):
            try:
                files.add(importlib.util.cache_from_source(f))
            except Exception:  # noqa: BLE001
                pass
    try:
        with open("/proc/self/maps") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 6 and parts[5].startswith("/"):
                    files.add(parts[5])
    except OSError:
        pass
    keep = sorted(f for f in files if f.startswith(("/lus/", "/home/", "/ec/", "/perm/", "/hpcperm/", "/etc/ecmwf/")) and os.path.isfile(f))
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "w") as fh:
        fh.write("\n".join(keep) + "\n")
    os.replace(tmp, path)
    LOG.warning("SPEED manifest: %d files written to %s", len(keep), path)


def _patch_cond_layernorm():
    """SPEED_LNROW=1: ConditionalLayerNorm computes its scale and bias from ONE conditioning row and broadcasts.

    The noise conditioning is one 16-vector per (batch, ensemble) element, repeated for every node; inference runs batch 1,
    ensemble 1, so all rows of `cond` are identical and Linear(16 -> 512) over 6.6 M (data) or 0.66 M (hidden) identical rows
    recomputes the same 512 numbers per node (two GEMMs and two full-size temporaries per layer norm, 40 layer norms per
    call; 30 % of a TF32 call in the operator profile of job 38576955). Same algebra, float32 round-off differences only.
    The first call of every module checks that the rows really are identical and refuses otherwise."""
    from anemoi.models.layers.normalization import ConditionalLayerNorm

    if getattr(ConditionalLayerNorm, "_speed_lnrow", False):
        return
    checked = set()

    def forward(self, x, cond):
        if id(self) not in checked:
            if cond.shape[0] > 1 and not bool((cond == cond[:1]).all()):
                raise RuntimeError("SPEED_LNROW: conditioning rows differ; the one-row shortcut does not apply")
            checked.add(id(self))
        c = cond[:1]
        scale = self.scale(c)
        bias = self.bias(c)
        out = self.norm(x)
        out = out * (scale + 1.0) + bias
        return out.type_as(x) if self.autocast else out

    ConditionalLayerNorm.forward = forward
    ConditionalLayerNorm._speed_lnrow = True
    LOG.warning("SPEED ConditionalLayerNorm: one conditioning row, broadcast")


def _patch_edge_pad(k_pad):
    """SPEED_EDGEPAD=<k>: the edge projection lin_edge maps the few static edge attributes (inner dimension 3) to 512 per edge.
    An inner dimension of 3 cannot use the TF32 tensor cores, so cuBLAS runs a plain `ampere_sgemm` (20 % of a call after
    SPEED_LNROW, profile of job 38579706). Zero-padding the attributes and the weight columns to k_pad changes no product
    (each added term is exactly 0 * 0) and lets the GEMM use tensor cores. The padded weight is built once per module."""
    import torch.nn.functional as F
    from anemoi.models.layers.block import GraphTransformerBaseBlock

    if getattr(GraphTransformerBaseBlock, "_speed_edgepad", False):
        return
    orig = GraphTransformerBaseBlock.get_qkve
    seen = set()

    def get_qkve(self, x, edge_attr):
        if getattr(self, "edge_spectral_reparam", False) or not isinstance(self.edge_pre_mlp, torch.nn.Identity):
            return orig(self, x, edge_attr)
        k = edge_attr.shape[-1]
        if k >= k_pad:
            return orig(self, x, edge_attr)
        x_src, x_dst = x if isinstance(x, tuple) else (x, x)
        query, key, value = self.lin_query(x_dst), self.lin_key(x_src), self.lin_value(x_src)
        w = self.lin_edge.weight
        wp = getattr(self, "_speed_wpad", None)
        if wp is None or wp.dtype != w.dtype or wp.device != w.device:
            wp = torch.zeros(w.shape[0], k_pad, dtype=w.dtype, device=w.device)
            wp[:, :k] = w.detach()
            self._speed_wpad = wp
        ea = F.pad(edge_attr, (0, k_pad - k))
        edges = F.linear(ea, wp, self.lin_edge.bias)
        if id(self) not in seen:
            seen.add(id(self))
            LOG.warning("SPEED edge pad: %s edges %s inner dim %d -> %d", type(self).__name__, tuple(edge_attr.shape), k, k_pad)
        return query, key, value, edges

    GraphTransformerBaseBlock.get_qkve = get_qkve
    GraphTransformerBaseBlock._speed_edgepad = True
    LOG.warning("SPEED GraphTransformer lin_edge: inputs zero-padded to %d columns", k_pad)


def _install_torchprof(inner):
    """SPEED_TORCHPROF=<call index>: run torch.profiler over that single denoiser call of the first draw and log the top
    ops by self CUDA time (rank 0 writes <cwd>/torchprof_rank0.txt)."""
    at = int(os.environ["SPEED_TORCHPROF"])
    base = inner.fwd_with_preconditioning
    st = {"i": 0, "done": False}

    def profiled(*a, **k):
        st["i"] += 1
        if st["done"] or st["i"] != at:
            return base(*a, **k)
        from torch.profiler import ProfilerActivity, profile, record_function

        st["done"] = True
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("denoiser_call"):
                out = base(*a, **k)
            torch.cuda.synchronize()
        try:
            tab = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=45, max_name_column_width=70)
            tab += "\n\n# grouped by input shape (ops only)\n" + prof.key_averages(group_by_input_shape=True).table(
                sort_by="self_cuda_time_total", row_limit=40, max_name_column_width=40, max_shapes_column_width=110)
            rank = int(os.environ.get("SLURM_PROCID", "0"))
            if rank == 0:
                with open("torchprof_rank0.txt", "w") as fh:
                    fh.write(tab)
                LOG.warning("SPEED torch profiler, call %d, rank 0:\n%s", at, tab)
        except Exception as e:  # noqa: BLE001
            LOG.warning("SPEED torch profiler table failed: %s", e)
        return out

    inner.fwd_with_preconditioning = profiled


def _install_probe(inner, variants):
    """On-path single-call precision probe. During the draw of lead SPEED_PROBE_LEAD (hours), at the denoiser calls listed in
    SPEED_PROBE_CALLS (1-based), the call is run in float32 (reference, returned to the sampler, so the trajectory stays the
    float32 one) and again under every variant; each rank logs, per variant, the RMS of (variant - reference) per output
    variable, the same divided by sigma (the unaccounted noise the variant injects at that step), a whiteness index
    (RMS of the first difference along the point index / (sqrt(2) RMS): 1 = white, << 1 = smooth), seconds and peak memory."""
    calls = {int(c) for c in os.environ["SPEED_PROBE_CALLS"].split(",") if c.strip()}
    base = inner.fwd_with_preconditioning
    state = {"i": 0, "on": False}
    inner._speed_probe_state = state
    amp_of = {"fp32": None, "tf32": None, "bf16": torch.bfloat16, "fp16": torch.float16, "tf32bf16": torch.bfloat16}

    def _run(amp, guards, a, k, amps=()):
        FP32.ACTIVE = set(inner._speed_static_guards) | set(guards)
        AMP16.ACTIVE = set(inner._speed_static_amps) | set(amps)
        torch.backends.cuda.matmul.allow_tf32 = amp in ("tf32", "tf32bf16")
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=amp_of[amp] or torch.bfloat16, enabled=amp_of[amp] is not None):
            out = base(*a, **k)
        torch.cuda.synchronize()
        return out, time.perf_counter() - t, torch.cuda.max_memory_allocated() / 2**30

    def probed(*a, **k):
        state["i"] += 1
        if not state["on"] or state["i"] not in calls:
            return base(*a, **k)
        tf32_before, active_before = torch.backends.cuda.matmul.allow_tf32, set(FP32.ACTIVE)
        sig = float(next(iter(a[2].values())).reshape(-1)[0])
        ref, t_ref, m_ref = _run("fp32", [], a, k, ())
        r = ref["out_hres"].double()
        rr = r.reshape(-1, r.shape[-1])
        LOG.warning("SPEEDPROBE call=%d sigma=%.5g variant=REF_fp32 seconds=%.3f peak_GiB=%.2f rms_D=%s", state["i"], sig, t_ref,
                    m_ref, [float("%.4g" % v) for v in rr.pow(2).mean(0).sqrt().tolist()])
        amps_before = set(AMP16.ACTIVE)
        for tag, amp, guards, amps in variants:
            try:
                alt, t_alt, m_alt = _run(amp, guards, a, k, amps)
                d = (alt["out_hres"].double() - r).reshape(-1, r.shape[-1])
                nbad = int((~torch.isfinite(d)).sum())
                rms = d.pow(2).mean(0).sqrt()
                hp = (d[1:] - d[:-1]).pow(2).mean(0).sqrt() / (rms * 2**0.5 + 1e-300)
                bias = d.mean(0)
                LOG.warning("SPEEDPROBE call=%d sigma=%.5g variant=%s seconds=%.3f peak_GiB=%.2f nonfinite=%d rms_diff=%s "
                            "rms_diff_over_sigma=%s whiteness=%s mean_diff=%s", state["i"], sig, tag, t_alt, m_alt, nbad,
                            [float("%.3g" % v) for v in rms.tolist()], [float("%.3g" % (v / sig)) for v in rms.tolist()],
                            [float("%.2f" % v) for v in hp.tolist()], [float("%.2g" % v) for v in bias.tolist()])
                del alt, d
            except Exception as e:  # noqa: BLE001
                LOG.warning("SPEEDPROBE call=%d sigma=%.5g variant=%s FAILED %s: %s", state["i"], sig, tag, type(e).__name__,
                            str(e)[:300])
                torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = tf32_before
        FP32.ACTIVE = active_before
        AMP16.ACTIVE = amps_before
        return ref

    inner.fwd_with_preconditioning = probed
    LOG.warning("SPEED probe installed: calls %s, lead %s, variants %s", sorted(calls), os.environ.get("SPEED_PROBE_LEAD", "24"),
                variants)


def _patch():
    import downscaling_unified_runner as dur

    cls = dur.UnifiedDownscalingRunner
    if getattr(cls, "_speed_patched", False):
        return
    if os.environ.get("SPEED_TF32", "") == "1":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        LOG.warning("SPEED TF32 matmul enabled")
    orig = cls.predict_step

    def predict_step(self, model, input_tensor_torch, *args, **kwargs):
        inner = _prepare(model)
        pstate = getattr(inner, "_speed_probe_state", None)
        if pstate is not None:
            step_ = args[1] if len(args) >= 2 else kwargs["step"]
            pstate["i"] = 0
            pstate["on"] = int(step_.total_seconds() // 3600) == int(os.environ.get("SPEED_PROBE_LEAD", "24"))
        acc = getattr(inner, "_speed_acc", None)
        if acc is not None:
            acc.clear()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = orig(self, model, input_tensor_torch, *args, **kwargs)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        calls = getattr(inner, "_gl512_calls", None)
        n = calls[0] if calls else -1
        rank = int(getattr(self, "global_rank", 0) or 0)
        LOG.warning(
            "SPEED draw rank=%d seconds=%.2f calls=%d s_per_call=%.4f peak_alloc_GiB=%.2f peak_reserved_GiB=%.2f "
            "amp=%s tf32=%s tf32_env=%s guards=%s profile=%s",
            rank, dt, n, dt / max(n, 1), torch.cuda.max_memory_allocated() / 2**30,
            torch.cuda.max_memory_reserved() / 2**30, os.environ.get("CODEX_UNIFIED_FORWARD_DTYPE", "fp32"),
            torch.backends.cuda.matmul.allow_tf32, os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", ""),
            os.environ.get("SPEED_FP32_MODULES", ""), {k: round(v, 2) for k, v in (acc or {}).items()},
        )
        man = os.environ.get("SPEED_MANIFEST", "")
        if man and rank == 0 and not os.path.exists(man):
            _write_manifest(man)
        dump = os.environ.get("SPEED_DUMP_NPY", "")
        if dump and rank == 0:
            import numpy as np

            step = args[1] if len(args) >= 2 else kwargs["step"]
            os.makedirs(dump, exist_ok=True)
            np.save(os.path.join(dump, "out_lead%03d.npy" % int(step.total_seconds() // 3600)),
                    out.detach().float().cpu().numpy())
        return out

    cls.predict_step = predict_step
    cls._speed_patched = True
    LOG.warning("speed_patches: predict_step timed")


_patch()
