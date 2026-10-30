"""CPU tests of the health audit on a small GraphTransformer-shaped model saved like a Lightning checkpoint.

The toy model has the anemoi state-dict names (lin_query ... projection, q_norm, node_dst_mlp.0/.2,
layer_norm_mlp_dst, layer_norm_attention_dest alias) and three planted faults whose counts the audit must return:
a zero-content head and zero-content MLP units (the widening saddle), gradient-dead units with nonzero weights
(bias pushed to -100: GELU gradient exactly 0, the DB300 case), and a zero LayerNorm gain channel.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

C, H, D, HID, N = 32, 4, 8, 64, 40
ZERO_HEAD = 3
ZERO_UNITS = [60, 61, 62, 63]
GRAD_DEAD = [5, 6, 7]
ZERO_GAIN_CH = 2


class GTBlock(nn.Module):
    def __init__(self, dest_alias=False):
        super().__init__()
        self.num_heads = H
        for n in ("lin_query", "lin_key", "lin_value", "lin_self"):
            setattr(self, n, nn.Linear(C, C))
        self.lin_edge = nn.Linear(3, C)
        self.projection = nn.Linear(C, C)
        self.q_norm = nn.LayerNorm(D, bias=False)
        self.k_norm = nn.LayerNorm(D, bias=False)
        self.layer_norm_attention = nn.LayerNorm(C)
        if dest_alias:
            self.layer_norm_attention_dest = self.layer_norm_attention
        self.layer_norm_mlp_dst = nn.LayerNorm(C)
        self.node_dst_mlp = nn.Sequential(nn.Linear(C, HID), nn.GELU(), nn.Linear(HID, C))

    def forward(self, x, edge_attr, edge_index):
        xn = self.layer_norm_attention(x)
        q = self.q_norm(self.lin_query(xn).view(-1, H, D))
        k = self.k_norm(self.lin_key(xn).view(-1, H, D))
        v = self.lin_value(xn).view(-1, H, D)
        e = self.lin_edge(edge_attr).view(-1, H, D)
        src, dst = edge_index
        logit = (q[dst] * (k[src] + e)).sum(-1) / D**0.5
        w = torch.exp(logit - logit.max())
        den = torch.zeros(x.shape[0], H).index_add_(0, dst, w)
        msg = torch.zeros(x.shape[0], H, D).index_add_(0, dst, (w / den[dst]).unsqueeze(-1) * (v[src] + e))
        out = self.projection(msg.reshape(-1, C) + self.lin_self(xn)) + x
        return out + self.node_dst_mlp(self.layer_norm_mlp_dst(out)), edge_attr


class Proc(nn.Module):
    def __init__(self):
        super().__init__()
        self.proc = nn.ModuleList([GTBlock(), GTBlock()])


class Wrap(nn.Module):
    use_checkpoint = False

    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.proc = GTBlock(dest_alias=True)
        self.processor = Proc()
        self.decoder = nn.Module()
        self.decoder.proc = GTBlock()

    def forward(self, x, ea, ei):
        x, _ = self.encoder.proc(x, ea, ei)
        for b in self.processor.proc:
            if self.use_checkpoint:
                x, _ = torch.utils.checkpoint.checkpoint(b, x, ea, ei, use_reentrant=False)
            else:
                x, _ = b(x, ea, ei)
        return self.decoder.proc(x, ea, ei)[0]


class Interface(nn.Module):  # model.model.<...> like AnemoiModelInterface inside the Lightning module
    def __init__(self):
        super().__init__()
        self.model = Wrap()


class Normalizer(nn.Module):  # anemoi's pre/post processors hold float buffers in the state dict
    def __init__(self, n):
        super().__init__()
        self.register_buffer("_norm_mul", torch.ones(n))
        self.register_buffer("_norm_add", torch.zeros(n))


class Lit(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Interface()
        # same width as the biases (C): must never be taken for a parameter by the optimizer mapping
        self.model.pre_processors = nn.Module()
        self.model.pre_processors.normalizer = Normalizer(C)


def _graph(seed=0):
    g = torch.Generator().manual_seed(seed)
    ei = torch.stack([torch.randint(0, N, (200,), generator=g), torch.arange(200) % N])
    return torch.randn(N, C, generator=g), torch.randn(200, 3, generator=g), ei


def _plant(lit: Lit):
    b = lit.model.model.processor.proc[1]
    with torch.no_grad():
        rows = slice(ZERO_HEAD * D, (ZERO_HEAD + 1) * D)
        for n in ("lin_query", "lin_key", "lin_value", "lin_self", "lin_edge"):
            getattr(b, n).weight[rows] = 0
            getattr(b, n).bias[rows] = 0
        b.projection.weight[:, rows] = 0
        mlp = b.node_dst_mlp
        mlp[0].weight[ZERO_UNITS] = 0
        mlp[0].bias[ZERO_UNITS] = 0
        mlp[2].weight[:, ZERO_UNITS] = 0
        mlp[0].bias[GRAD_DEAD] = -100.0  # GELU output and gradient exactly 0 on every node


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    torch.manual_seed(0)
    lit = Lit()
    _plant(lit)
    opt = torch.optim.AdamW(lit.parameters(), lr=1e-3, weight_decay=0)
    x, ea, ei = _graph()
    for _ in range(3):
        opt.zero_grad()
        lit.model.model(x, ea, ei).pow(2).mean().backward()
        opt.step()
    with torch.no_grad():  # a zero gain gets a gradient, so plant it after the steps
        lit.model.model.processor.proc[1].layer_norm_mlp_dst.weight[ZERO_GAIN_CH] = 0
    shadow = {k[len("model."):]: v.detach().clone() for k, v in lit.state_dict().items()}
    obj = {
        "global_step": 3000,
        "state_dict": lit.state_dict(),
        "optimizer_states": [opt.state_dict()],
        "callbacks": {"EMA": {"shadows": {10000: shadow}}},
    }
    p = tmp_path_factory.mktemp("ck") / "anemoi-by_step-epoch_000-step_003000.ckpt"
    torch.save(obj, p)
    return p, lit


def test_static_counts(ckpt):
    from health.static_checks import capacity_summary, run_static

    p, _ = ckpt
    r, blocks, checks = run_static(str(p))
    roles = [b.role for b in blocks]
    assert roles == ["encoder", "processor.0", "processor.1", "decoder"]
    assert all(b.heads == H and b.head_dim == D for b in blocks)
    z = checks["c1_zero_content"]
    assert z["indices"]["processor.1"]["zero_heads"] == [ZERO_HEAD]
    assert z["indices"]["processor.1"]["node_dst_mlp.zero_units"] == ZERO_UNITS
    assert z["indices"]["processor.0"]["zero_heads"] == []
    assert z["nonfinite"]["model"]["values"] == 0
    assert z["alias_mismatch"]["model"] == {"pairs": 2, "mismatched": 0}
    o = checks["c3_optimizer_dead"]
    assert o["optimizer"] == "adamw" and o["mapping_complete"], o
    assert o["indices"]["processor.1"]["node_dst_mlp.grad_dead_units"] == GRAD_DEAD
    assert o["indices"]["processor.0"]["node_dst_mlp.grad_dead_units"] == []
    assert o["agreement"]["processor.1"]["node_dst_mlp"]["all"] == len(GRAD_DEAD) + len(ZERO_UNITS)
    s = checks["c6_scale"]["per_role"]["processor.1"]
    assert s["layer_norm_mlp_dst.zero_gain_frac"] == pytest.approx(1 / C)
    assert s["edge_logit_scale.max"] > 0
    cap = capacity_summary(blocks, checks)
    assert cap["units_dead_zero"] == len(ZERO_UNITS) and cap["units_dead_grad"] == len(GRAD_DEAD)
    assert cap["heads_dead_zero"] == 1


def test_cli_audit_and_diff(ckpt, tmp_path, capsys):
    from health.cli import main

    p, _ = ckpt
    assert main(["audit", str(p), "--out", str(tmp_path / "a"), "--id", "toy"]) == 0
    out = capsys.readouterr().out
    assert "processor MLP units dead" in out and "FAIL" in out
    audit = json.loads((tmp_path / "a" / "audit.json").read_text())
    assert audit["capacity"]["processor"]["unit_dead_frac"] == pytest.approx(7 / (2 * HID))
    assert any("python -m health revive" in c for c in audit["revive"])
    fails = {r["metric"] for r in audit["bands"] if r["status"] == "FAIL"}
    assert "c3_optimizer_dead.processor.1.node_dst_mlp.grad_dead_unit_frac" in fails
    assert "c1_zero_content.processor.1.zero_head_frac" in fails
    assert main(["audit", str(p), "--out", str(tmp_path / "b"), "--prev", str(tmp_path / "a" / "audit.json")]) == 0
    b = json.loads((tmp_path / "b" / "audit.json").read_text())
    assert b["diffs"]["previous"]["moved"] == []  # same file, nothing moves
    assert main(["diff", str(tmp_path / "a" / "audit.json"), str(tmp_path / "b" / "audit.json")]) == 0


def test_revive_roundtrip(ckpt, tmp_path):
    from health.cli import main
    from health.static_checks import run_static

    p, _ = ckpt
    main(["audit", str(p), "--out", str(tmp_path / "a"), "--id", "toy"])
    # without check-4 / 3b evidence the gradient-dead units are skipped (their output is not proven zero)
    audit = json.loads((tmp_path / "a" / "audit.json").read_text())
    audit["checks"].pop("c3b_saturation_bound")
    (tmp_path / "no_evidence.json").write_text(json.dumps(audit))
    safe = tmp_path / "safe.ckpt"
    assert main(["revive", str(p), str(safe), "--audit", str(tmp_path / "no_evidence.json")]) == 0
    w2 = torch.load(safe, weights_only=False)["state_dict"]["model.model.processor.proc.1.node_dst_mlp.2.weight"]
    assert torch.all(w2[:, ZERO_UNITS] == 0) and torch.all(w2[:, GRAD_DEAD] != 0)
    dst = tmp_path / "revived.ckpt"
    assert main(["revive", str(p), str(dst), "--audit", str(tmp_path / "a" / "audit.json"), "--unproven"]) == 0
    _, _, checks = run_static(str(dst))
    z = checks["c1_zero_content"]["indices"]["processor.1"]
    assert z["zero_heads"] == [] and z["node_dst_mlp.zero_units"] == []
    ck = torch.load(dst, weights_only=False)
    sd = ck["state_dict"]
    w2 = sd["model.model.processor.proc.1.node_dst_mlp.2.weight"]
    assert torch.all(w2[:, ZERO_UNITS + GRAD_DEAD] == 0)  # output side zero: function unchanged
    assert "optimizer_states" not in ck
    shadow = ck["callbacks"]["EMA"]["shadows"][10000]
    assert torch.equal(shadow["model.processor.proc.1.node_dst_mlp.0.weight"], sd["model.model.processor.proc.1.node_dst_mlp.0.weight"])


def _toy_cases(root):
    from health.forward_checks import Case, CaseResult

    cases = []
    for seed in range(4):
        x0, ea, ei = _graph(seed)

        def run(input_grad, perturb, seed, x0=x0, ea=ea, ei=ei):
            x = x0.clone() + (perturb * torch.randn_like(x0) if perturb else 0)
            x.requires_grad_(input_grad)
            y = root.model(x, ea, ei)
            return CaseResult(loss=y.pow(2).mean(), output=y, inputs={"x": x}, output_vars=[f"v{i}" for i in range(C)])

        cases.append(Case(label=f"toy{seed}", sigma=1.0, run=run))
    return cases


def test_forward_identical_with_gradient_checkpointing(ckpt):
    """anemoi checkpoints its blocks by default: hooks fire twice per forward. Results must not change."""
    from health.forward_checks import run_grad_pass

    _, lit = ckpt
    root = lit.model
    plain = run_grad_pass(root, _toy_cases(root), "cpu")
    Wrap.use_checkpoint = True
    try:
        ck = run_grad_pass(root, _toy_cases(root), "cpu")
    finally:
        Wrap.use_checkpoint = False
    assert ck["c2_grad_dead"]["indices"] == plain["c2_grad_dead"]["indices"]
    for cid in ("c4_preact", "c7_stream"):
        for role, m in plain[cid]["per_role"].items():
            for k, v in m.items():
                assert ck[cid]["per_role"][role][k] == pytest.approx(v, rel=1e-6), (cid, role, k)


def test_forward_engine(ckpt):
    from health.forward_checks import Case, CaseResult, run_flow, run_grad_pass

    _, lit = ckpt
    root = lit.model  # like AnemoiModelInterface: state-dict names start with model.
    cases = []
    for seed in range(4):
        x0, ea, ei = _graph(seed)

        def run(input_grad, perturb, seed, x0=x0, ea=ea, ei=ei):
            x = x0.clone() + (perturb * torch.randn_like(x0) if perturb else 0)
            x.requires_grad_(input_grad)
            y = root.model(x, ea, ei)
            return CaseResult(loss=y.pow(2).mean(), output=y, inputs={"x": x}, output_vars=[f"v{i}" for i in range(C)])

        cases.append(Case(label=f"toy{seed}", sigma=1.0, run=run))
    res = run_grad_pass(root, cases, "cpu")
    i2 = res["c2_grad_dead"]["indices"]["processor.1"]
    assert set(GRAD_DEAD) | set(ZERO_UNITS) <= set(i2["node_dst_mlp.dead_units"])
    assert ZERO_HEAD in i2["dead_heads"]
    assert res["c2_grad_dead"]["indices"]["processor.0"]["dead_heads"] == []
    m4 = res["c4_preact"]["per_role"]["processor.1"]
    assert m4["node_dst_mlp.no_grad_frac"] == pytest.approx(len(GRAD_DEAD) / HID)
    m7 = res["c7_stream"]["per_role"]["processor.0"]
    assert 0 < m7["update_ratio"] and "dirichlet" in m7 and "mlp_ratio" in m7
    flow = run_flow(root, cases[:2])["c9_flow"]
    assert flow["per_role"]["model"]["unread_input_frac"] == 0
    assert flow["per_role"]["model"]["silent_output_frac"] == 0


def test_graph_audit():
    from health.forward_checks import graph_audit

    class Store(dict):
        __getattr__ = dict.__getitem__

    class Nodes:
        def __init__(self, n):
            self.num_nodes = n

    class G:
        edge_types = [("hidden", "to", "hidden"), ("data", "to", "hidden")]

        def __getitem__(self, k):
            if k == "hidden":
                return Nodes(5)
            if k == "data":
                return Nodes(3)
            if k == ("hidden", "to", "hidden"):
                return Store(edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 0, 3]]), edge_length=torch.rand(4, 1))
            return Store(edge_index=torch.tensor([[0, 1, 2], [0, 1, 1]]))

    out = graph_audit(G())["c8_graph"]["per_role"]
    hh = out["graph:hidden_to_hidden"]
    assert hh["isolated_frac"] == pytest.approx(1 / 5)  # node 4 has no in-edge
    assert hh["extra_components"] == 2  # {0,1,2}, {3}, {4}
    assert out["graph:data_to_hidden"]["isolated_frac"] == pytest.approx(3 / 5)


def test_bands_cover_metrics():
    from health import report

    rules = report.load_bands()
    assert report.status(0.0, report.rule_for("c3_optimizer_dead.processor.node_dst_mlp.grad_dead_unit_frac", rules)) == "ok"
    assert report.status(0.01, report.rule_for("c3_optimizer_dead.processor.node_dst_mlp.grad_dead_unit_frac", rules)) == "FAIL"
    assert report.status(0.005, report.rule_for("c7_stream.processor.update_ratio", rules)) == "WARN"
    assert report.status(3.0, report.rule_for("c6_scale.processor.lin_query.rms_over_init", rules)) == "info"
    assert np.isfinite(0.0)


def test_optimizer_map_two_groups_and_frozen(tmp_path):
    """Muon-style split (matrices in group 0, the rest in group 1) and frozen QK gains (no optimizer entry)."""
    from health.static_checks import run_static

    torch.manual_seed(1)
    lit = Lit()
    _plant(lit)
    for n, prm in lit.named_parameters():
        if n.endswith(("q_norm.weight", "k_norm.weight")):
            prm.requires_grad_(False)
    mats = [p for p in lit.parameters() if p.requires_grad and p.dim() == 2]
    rest = [p for p in lit.parameters() if p.requires_grad and p.dim() != 2]
    opt = torch.optim.AdamW([{"params": mats}, {"params": rest, "lr": 5e-4}], lr=1e-3, weight_decay=0)
    x, ea, ei = _graph()
    for _ in range(2):
        opt.zero_grad()
        lit.model.model(x, ea, ei).pow(2).mean().backward()
        opt.step()
    p = tmp_path / "two_groups.ckpt"
    torch.save({"global_step": 2, "state_dict": lit.state_dict(), "optimizer_states": [opt.state_dict()]}, p)
    _, _, checks = run_static(str(p))
    o = checks["c3_optimizer_dead"]
    assert o["mapping_complete"], o
    assert o["indices"]["processor.1"]["node_dst_mlp.grad_dead_units"] == GRAD_DEAD
    assert o["agreement"]["processor.1"]["node_dst_mlp"]["all"] == len(GRAD_DEAD) + len(ZERO_UNITS)
    assert o["mapping"] == "exact"
    assert all(k.endswith(("q_norm.weight", "k_norm.weight")) for k, _ in o["unmapped"])


def _add_metadata(path, config):
    """Append anemoi-metadata/anemoi.json to a torch zip, as anemoi does."""
    import zipfile

    with zipfile.ZipFile(path) as zf:
        root = zf.namelist()[0].split("/")[0]
    with zipfile.ZipFile(path, "a") as zf:
        zf.writestr(f"{root}/anemoi-metadata/anemoi.json", json.dumps({"config": config}))


def _train(lit, groups, steps=2):
    opt = torch.optim.AdamW(groups, lr=1e-3, weight_decay=0, betas=(0.9, 0.95))
    x, ea, ei = _graph()
    for _ in range(steps):
        opt.zero_grad()
        lit.model.model(x, ea, ei).pow(2).mean().backward()
        opt.step()
    return opt


def test_frozen_block_is_ambiguous_without_config_and_exact_with_it(tmp_path):
    """A frozen processor block shares every shape with the next block: without the config the mapping must say
    AMBIGUOUS; with training.submodules_to_freeze naming it, the block is excluded and the mapping is exact."""
    from health.static_checks import run_static

    torch.manual_seed(2)
    lit = Lit()
    _plant(lit)
    for prm in lit.model.model.processor.proc[0].parameters():
        prm.requires_grad_(False)
    opt = _train(lit, [p for p in lit.parameters() if p.requires_grad])
    p = tmp_path / "frozen_block.ckpt"
    torch.save({"global_step": 2, "state_dict": lit.state_dict(), "optimizer_states": [opt.state_dict()]}, p)
    _, _, checks = run_static(str(p))
    assert checks["c3_optimizer_dead"]["mapping"] == "AMBIGUOUS"
    _add_metadata(p, {"training": {"submodules_to_freeze": ["proc.0"]}})
    _, _, checks = run_static(str(p))
    o = checks["c3_optimizer_dead"]
    assert o["mapping"] == "exact", o["unmapped"]
    assert o["indices"]["processor.1"]["node_dst_mlp.grad_dead_units"] == GRAD_DEAD
    assert o["window_min_steps"] and 1000 < o["window_min_steps"] < 3000  # beta2 0.95: ~1.5k steps


def test_soap_uses_gg_diagonals(tmp_path):
    """SOAP: exp_avg_sq of matrices lives in a rotated basis; the per-unit record is diag(GG). Rotate exp_avg_sq to
    make it useless per unit and check the dead units still come out of GG."""
    from health.static_checks import run_static

    torch.manual_seed(3)
    lit = Lit()
    _plant(lit)
    opt = _train(lit, list(lit.parameters()), steps=1)
    x, ea, ei = _graph()
    opt.zero_grad()
    lit.model.model(x, ea, ei).pow(2).mean().backward()
    sd = opt.state_dict()
    params = list(lit.parameters())
    for i, prm in enumerate(params):
        s = sd["state"][i]
        g = prm.grad.detach()
        if g.dim() == 2:
            s["GG"] = [g @ g.T, g.T @ g]
            q, _ = torch.linalg.qr(torch.randn(g.shape[0], g.shape[0]))
            s["exp_avg_sq"] = (q @ s["exp_avg_sq"]).abs() + 1e-9  # rotated: no zero rows left
        else:
            s["GG"] = [[]]
        s["Q"] = None
    p = tmp_path / "soap.ckpt"
    torch.save({"global_step": 2, "state_dict": lit.state_dict(), "optimizer_states": [sd]}, p)
    _, _, checks = run_static(str(p))
    o = checks["c3_optimizer_dead"]
    assert o["optimizer"] == "soap"
    assert o["indices"]["processor.1"]["node_dst_mlp.grad_dead_units"] == GRAD_DEAD
    assert set(o["agreement"]["processor.1"]["node_dst_mlp"]) == {"bias", "w1_row", "w2_col", "all"}


def test_saturation_bound_proves_planted_units(ckpt, tmp_path):
    """Bias -100 with |W1 row| ~ 0.2 and plain LayerNorm gains ~1: UB = -100 + sqrt(C) |W1 row| << -14."""
    from health.cli import main
    from health.static_checks import run_static

    p, _ = ckpt
    _, _, checks = run_static(str(p))
    sb = checks["c3b_saturation_bound"]
    assert set(GRAD_DEAD) <= set(sb["indices"]["processor.1"]["node_dst_mlp.never_fires_units"])
    assert sb["indices"]["processor.0"]["node_dst_mlp.never_fires_units"] == []
    # with that CPU evidence, revive takes the gradient-dead units without --unproven
    main(["audit", str(p), "--out", str(tmp_path / "a"), "--id", "toy"])
    dst = tmp_path / "r.ckpt"
    assert main(["revive", str(p), str(dst), "--audit", str(tmp_path / "a" / "audit.json")]) == 0
    w2 = torch.load(dst, weights_only=False)["state_dict"]["model.model.processor.proc.1.node_dst_mlp.2.weight"]
    assert torch.all(w2[:, GRAD_DEAD + ZERO_UNITS] == 0)


def test_plain_unwraps_omegaconf_stubs():
    from health.ckpt import Stub, plain

    def node(v):
        n = Stub()
        n.__setstate__({"_val": v, "_metadata": None})
        return n

    lc = Stub()
    lc.__setstate__({"_content": [node(0.9), node(0.95)], "_metadata": None})
    assert plain(lc) == [0.9, 0.95]
    assert plain({"betas": lc, "lr": 1e-4}) == {"betas": [0.9, 0.95], "lr": 1e-4}
