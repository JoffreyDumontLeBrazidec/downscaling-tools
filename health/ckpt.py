"""Torch-free reader for torch zip checkpoints (Lightning training files and pickled-module inference files).

numpy only, so the CPU checks run on any login node. Tensors come back as lazy handles; ``Reader.array(t)``
reads one storage member from the zip. A 1.8 GB checkpoint opens in seconds; reading every tensor takes ~30 s.

Consolidates the three copies in the docs kits (``m2/ckread.py``, ``audit_dead_units.py``,
``adam_state_dead_units.py``); the unpickler is the most permissive of them (``ckread.py``: pickled modules,
DotDict items, ``_rebuild_parameter_with_state``).
"""
from __future__ import annotations

import collections
import json
import os
import pickle
import zipfile
from dataclasses import dataclass, field

import numpy as np

_DTYPES = {
    "FloatStorage": "<f4",
    "DoubleStorage": "<f8",
    "HalfStorage": "<f2",
    "BFloat16Storage": "bf16",
    "LongStorage": "<i8",
    "IntStorage": "<i4",
    "ShortStorage": "<i2",
    "CharStorage": "<i1",
    "ByteStorage": "<u1",
    "BoolStorage": "?",
}


class _StubMeta(type):
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return Stub()


class Stub(metaclass=_StubMeta):
    """Stand-in for any non-builtin class in the pickle; keeps constructor args and pickled state."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.state = None

    def __setstate__(self, state):
        self.state = state

    def __getattr__(self, name):
        if name.startswith("__") or name in ("items_", "list_"):
            raise AttributeError(name)
        return Stub()

    def __setitem__(self, k, v):
        self.__dict__.setdefault("items_", {})[k] = v

    def append(self, v):
        self.__dict__.setdefault("list_", []).append(v)

    def extend(self, vs):
        for v in vs:
            self.append(v)

    def __call__(self, *args, **kwargs):
        return Stub(*args)


class LazyTensor:
    __slots__ = ("storage", "offset", "size", "stride")

    def __init__(self, storage, offset, size, stride):
        self.storage = storage  # (storage type name, key, location, numel)
        self.offset = int(offset)
        self.size = tuple(int(s) for s in size)
        self.stride = tuple(int(s) for s in stride)

    @property
    def shape(self):
        return self.size

    @property
    def numel(self):
        return int(np.prod(self.size)) if self.size else 1

    @property
    def is_float(self):
        return _DTYPES.get(self.storage[0], "") in ("<f4", "<f8", "<f2", "bf16")


def _rebuild_tensor_v2(storage, storage_offset, size, stride, *rest, **kw):
    return LazyTensor(storage, storage_offset, size, stride)


def _rebuild_parameter(data, *rest, **kw):
    return data


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch._utils" and name in ("_rebuild_parameter", "_rebuild_parameter_with_state"):
            return _rebuild_parameter
        if module == "torch._utils" and name.startswith("_rebuild"):
            return lambda *a, **k: None
        if module in ("builtins", "__builtin__", "copyreg", "copy_reg", "_codecs"):
            return pickle.Unpickler.find_class(self, module, name)
        return _StubMeta(str(name), (Stub,), {})

    def persistent_load(self, pid):
        stype = pid[1]
        sname = stype.__name__ if isinstance(stype, type) else str(stype)
        return (sname, str(pid[2]), pid[3], pid[4])


def _module_params(obj, prefix=""):
    """Walk a pickled nn.Module stub; yield (dotted name, LazyTensor) for _parameters and _buffers."""
    st = getattr(obj, "state", None)
    if not isinstance(st, dict):
        return
    for kind in ("_parameters", "_buffers"):
        for k, v in (st.get(kind) or {}).items():
            if isinstance(v, LazyTensor):
                yield prefix + k, v
    for k, m in (st.get("_modules") or {}).items():
        if m is not None:
            yield from _module_params(m, prefix + k + ".")


_BUFFER_SUFFIXES = (".frequencies", ".pi", ".muzero", ".norm_min_val", ".perm", ".edge_attr", ".edge_index_base")


def _never_a_parameter(key: str) -> bool:
    """Float buffers that sit in the state dict between parameters (never in an optimizer group)."""
    return ".latlons_" in key or key.endswith(_BUFFER_SUFFIXES)


@dataclass
class OptimizerView:
    """The optimizer state mapped to state-dict names. ``by_name[key]`` is that parameter's state dict."""

    by_name: dict = field(default_factory=dict)
    kind: str = "none"  # adamw | muon | soap | other | none
    n_states: int = 0
    n_mapped: int = 0
    n_entries: int = 0  # float state-dict tensors that could hold a state (buffers, aliases, config-frozen removed)
    skipped: list = field(default_factory=list)  # [(name, shape)] float tensors left without a state
    excluded_frozen: list = field(default_factory=list)  # names removed because the config freezes them
    status: str = "none"  # exact | AMBIGUOUS | incomplete | none
    groups: list = field(default_factory=list)  # [{"n": params, "lr": .., "weight_decay": .., "betas": .., "use_muon": ..}]

    @property
    def complete(self) -> bool:
        return self.status == "exact"


class Reader:
    """One checkpoint file. ``state`` is the model state dict (name -> LazyTensor)."""

    def __init__(self, path: str):
        self.path = path
        self.size_gb = os.path.getsize(path) / 1e9
        self.zf = zipfile.ZipFile(path)
        pkl = [n for n in self.zf.namelist() if n.endswith("/data.pkl") or n == "data.pkl"]
        if len(pkl) != 1:
            raise ValueError(f"{path}: expected one data.pkl in the zip, found {pkl}")
        self.prefix = pkl[0][: -len("data.pkl")]
        with self.zf.open(pkl[0]) as fh:
            self.obj = _Unpickler(fh).load()
        if isinstance(self.obj, dict) and "state_dict" in self.obj:
            self.kind = "training"
            self.state = collections.OrderedDict(
                (k, v) for k, v in self.obj["state_dict"].items() if isinstance(v, LazyTensor)
            )
        else:
            self.kind = "inference"
            self.state = collections.OrderedDict(_module_params(self.obj))
        self.global_step = self.obj.get("global_step") if isinstance(self.obj, dict) else None
        self.metadata = self._metadata()

    # ---- tensors -------------------------------------------------------------------------------------------
    def array(self, t: LazyTensor, dtype=np.float64) -> np.ndarray:
        raw = self.zf.read(f"{self.prefix}data/{t.storage[1]}")
        dt = _DTYPES[t.storage[0]]
        if dt == "bf16":
            flat = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        else:
            flat = np.frombuffer(raw, dtype=dt)
        if not t.size:
            return np.array(flat[t.offset], dtype=dtype)
        out = np.lib.stride_tricks.as_strided(
            flat[t.offset :], shape=t.size, strides=tuple(s * flat.itemsize for s in t.stride)
        )
        return np.array(out, dtype=dtype) if dtype is not None else np.array(out)

    def get(self, key: str, dtype=np.float64):
        t = self.state.get(key)
        return None if t is None else self.array(t, dtype)

    # ---- metadata ------------------------------------------------------------------------------------------
    def _metadata(self) -> dict:
        for n in self.zf.namelist():
            if n.endswith("anemoi-metadata/anemoi.json") or n.endswith("ai-models.json"):
                try:
                    return json.loads(self.zf.read(n).decode())
                except (ValueError, UnicodeDecodeError):
                    return {}
        return {}

    @property
    def config(self) -> dict:
        return self.metadata.get("config", {}) or {}

    @property
    def run_id(self):
        md = self.metadata
        return md.get("run_id") or (md.get("config", {}).get("training", {}) or {}).get("run_id")

    # ---- EMA ----------------------------------------------------------------------------------------------
    def ema_shadows(self) -> dict:
        """{label: {name: LazyTensor}} for every EMA shadow dict in the Lightning callback states."""
        out = {}
        if self.kind != "training":
            return out
        for cb_name, cb_state in (self.obj.get("callbacks") or {}).items():
            shadows = cb_state.get("shadows") if isinstance(cb_state, dict) else None
            if not isinstance(shadows, dict):
                continue
            for decay, sd in shadows.items():
                if isinstance(sd, dict):
                    out[f"ema{decay}"] = collections.OrderedDict(
                        (k, v) for k, v in sd.items() if isinstance(v, LazyTensor)
                    )
        return out

    # ---- optimizer ----------------------------------------------------------------------------------------
    def frozen_patterns(self) -> list:
        """Name fragments the training config freezes (no optimizer state): submodules_to_freeze, freeze_qk_gains."""
        c = self.config
        pats = [f".{m}." for m in ((c.get("training", {}) or {}).get("submodules_to_freeze") or [])]
        for sec in ("encoder", "processor", "decoder"):
            if ((c.get("model", {}) or {}).get(sec) or {}).get("freeze_qk_gains"):
                pats += [f".{sec}.@q_norm.weight", f".{sec}.@k_norm.weight"]
        return pats

    @staticmethod
    def _frozen(key: str, pats: list) -> bool:
        for p in pats:
            if "@" in p:
                sec, leaf = p.split("@")
                if sec in key and key.endswith(leaf):
                    return True
            elif p in key:
                return True
        return False

    def optimizer(self, trainable: list | None = None) -> OptimizerView:
        """Map the first optimizer's per-index state to state-dict names.

        Lightning keeps the state by integer index, in param-group order. Each group's indices are matched in
        order against the state-dict entries by shape; buffers, aliases and the tensors the config freezes are
        removed first. The mapping is EXACT when every remaining float tensor got a state (a bijection in order),
        or when no tensor left without a state shares its shape with any state (it cannot have shifted the
        alignment). Otherwise it is AMBIGUOUS: a frozen or never-updated tensor may have taken a same-shape
        neighbour's state, and every name after it may be shifted (a whole frozen block would shift by a block).
        """
        view = OptimizerView()
        opts = self.obj.get("optimizer_states") if isinstance(self.obj, dict) else None
        if not opts:
            return view
        o = opts[0]
        st = o.get("state", {}) if isinstance(o, dict) else {}
        groups = o.get("param_groups", []) if isinstance(o, dict) else []
        view.n_states = len(st)
        keys0 = sorted(st.keys(), key=int)
        leaf_names = set()
        for k in keys0[:50]:
            leaf_names |= set(st[k].keys()) if isinstance(st[k], dict) else set()
        if any(isinstance(g, dict) and "use_muon" in g for g in groups) or "momentum_buffer" in leaf_names:
            view.kind = "muon"
        elif "GG" in leaf_names or "Q" in leaf_names:
            view.kind = "soap"
        elif "exp_avg_sq" in leaf_names:
            view.kind = "adamw"
        else:
            view.kind = "other"
        for g in groups:
            if isinstance(g, dict):
                view.groups.append({k: g.get(k) for k in ("lr", "weight_decay", "betas", "use_muon", "shampoo_beta")} | {"n": len(g.get("params", []))})
        if trainable is not None:
            # exact candidate list from the loaded model: named_parameters() order, aliases dropped, buffers absent,
            # requires_grad as the loader left it (state-dict names carry the Lightning 'model.' prefix)
            keyed = {k: t for k, t in self.state.items()}
            entries = [(k, keyed[k]) for k in trainable if k in keyed]
        else:
            entries = [(k, t) for k, t in self.state.items() if t.is_float and not _never_a_parameter(k)]
        seen_storage = set()
        dedup = []
        for k, t in entries:  # aliased tensors (layer_norm_attention_dest = layer_norm_attention) share a storage
            sig = (t.storage[1], t.offset, t.size)
            if sig not in seen_storage:
                seen_storage.add(sig)
                dedup.append((k, t))
        entries = dedup
        pats = self.frozen_patterns()
        view.excluded_frozen = [k for k, _ in entries if self._frozen(k, pats)]
        entries = [(k, t) for k, t in entries if not self._frozen(k, pats)]
        view.n_entries = len(entries)
        order = [int(i) for g in groups for i in g.get("params", [])] if groups else [int(k) for k in keys0]
        for gi, g in enumerate(groups or [{"params": order}]):
            pos = 0
            for idx in g.get("params", []):
                s = st.get(idx, st.get(str(idx)))
                ref = None
                if isinstance(s, dict):
                    ref = next((s[n] for n in ("exp_avg_sq", "exp_avg", "momentum_buffer") if isinstance(s.get(n), LazyTensor)), None)
                if ref is None:
                    continue
                while pos < len(entries) and tuple(entries[pos][1].shape) != tuple(ref.shape):
                    pos += 1
                if pos >= len(entries):
                    break
                name = entries[pos][0]
                if name in view.by_name:  # a later group re-walks from the start; never assign a name twice
                    while pos < len(entries) and (entries[pos][0] in view.by_name or tuple(entries[pos][1].shape) != tuple(ref.shape)):
                        pos += 1
                    if pos >= len(entries):
                        break
                    name = entries[pos][0]
                view.by_name[name] = s
                pos += 1
        view.n_mapped = len(view.by_name)
        view.skipped = [(k, t.shape) for k, t in entries if k not in view.by_name]
        state_shapes = set()
        for s in st.values():
            if isinstance(s, dict):
                ref = next((s[n] for n in ("exp_avg_sq", "exp_avg", "momentum_buffer") if isinstance(s.get(n), LazyTensor)), None)
                if ref is not None:
                    state_shapes.add(tuple(ref.shape))
        if view.n_mapped != view.n_states:
            view.status = "incomplete"
        elif not view.skipped or not any(tuple(sh) in state_shapes for _, sh in view.skipped):
            view.status = "exact"
        else:
            view.status = "AMBIGUOUS"
        return view

    def moment(self, opt_state: dict, which: str):
        """The 'history' moment of one parameter: exp_avg_sq (AdamW/SOAP 1-D) or momentum_buffer (Muon matrices)."""
        if which == "history":
            for n in ("exp_avg_sq", "momentum_buffer"):
                if isinstance(opt_state.get(n), LazyTensor):
                    return n, self.array(opt_state[n])
        elif which == "recent":
            for n in ("exp_avg", "momentum_buffer"):
                if isinstance(opt_state.get(n), LazyTensor):
                    return n, self.array(opt_state[n])
        return None, None
