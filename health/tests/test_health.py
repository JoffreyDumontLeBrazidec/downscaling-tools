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
            x, _ = b(x, ea, ei)
        return self.decoder.proc(x, ea, ei)[0]


class Interface(nn.Module):  # model.model.<...> like AnemoiModelInterface inside the Lightning module
    def __init__(self):
        super().__init__()
        self.model = Wrap()


class Lit(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Interface()


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
    assert o["agreement"]["processor.1"]["node_dst_mlp"]["all3"] == len(GRAD_DEAD) + len(ZERO_UNITS)
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
    dst = tmp_path / "revived.ckpt"
    assert main(["revive", str(p), str(dst), "--audit", str(tmp_path / "a" / "audit.json")]) == 0
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
    assert o["agreement"]["processor.1"]["node_dst_mlp"]["all3"] == len(GRAD_DEAD) + len(ZERO_UNITS)
    assert all(k.endswith(("q_norm.weight", "k_norm.weight", "_dest.weight", "_dest.bias")) for k in o["unmapped_examples"])
