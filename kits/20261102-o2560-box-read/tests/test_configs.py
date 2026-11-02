import subprocess
import sys

from conftest import CFG_A, CFG_READ, KIT_DIR, need


def test_A_and_B_configs_agree_on_model_graph_data(tmp_path):
    r = subprocess.run([sys.executable, str(KIT_DIR / "scripts" / "compare_configs.py"), need(CFG_A), need(CFG_READ),
                        str(tmp_path / "diff.txt")], capture_output=True, text=True)
    assert r.returncode == 0 and "CONFIG_COMPARE_PASS" in r.stdout, r.stdout + r.stderr
