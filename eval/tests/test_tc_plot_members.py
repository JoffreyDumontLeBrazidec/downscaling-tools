from __future__ import annotations

from pathlib import Path

import numpy as np

from eval.tc import plot_members_tc as mod


def test_run_tc_member_plots_generates_expected_files(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        mod,
        "CASES",
        [
            {
                "name": "idalia",
                "lat": (11.0, 12.0),
                "lon": (-80.0, -79.0),
                "dates": [28],
                "time": 1,
                "msl_levels": np.linspace(985, 1015, 5),
                "wind_levels": np.linspace(0, 30, 5),
            }
        ],
    )

    class _FakeRetriever:
        def retrieve_all_data(self, analysis, expid_enfo_O320, expid_eefo_O96, list_expid_ml):
            # [days, members, lead_times, lat, lon]
            arr = np.zeros((1, 3, 3, 120, 130), dtype=np.float32)
            msl = {
                "OPER_O320_0001": arr + 1000.0,
                "ENFO_O320_0001": arr + 1001.0,
                "EEFO_O96_0001": arr + 999.0,
                list_expid_ml[0]: arr + 1002.0,
            }
            wind10m = {
                "OPER_O320_0001": arr + 10.0,
                "ENFO_O320_0001": arr + 11.0,
                "EEFO_O96_0001": arr + 9.0,
                list_expid_ml[0]: arr + 12.0,
            }
            return msl, wind10m

    monkeypatch.setattr(mod, "_build_retriever", lambda *args, **kwargs: _FakeRetriever())

    created: list[str] = []

    def _fake_plot_field_members(**kwargs):
        out_path = tmp_path / "tc_members" / kwargs["filename"]
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("ok", encoding="utf-8")
        created.append(str(out_path))

    monkeypatch.setattr(mod, "_plot_member_data_legacy", _fake_plot_field_members)

    generated = mod.run_tc_member_plots(
        expver="j24v",
        outdir=str(tmp_path),
        members=[0, 1, 2],
    )

    assert len(generated) == 2
    assert generated == created
    for path in generated:
        p = Path(path)
        assert p.exists()
        assert p.parent == tmp_path / "tc_members"
