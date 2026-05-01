import subprocess
import sys
from pathlib import Path


def test_ci_regression_gate_rejects_bad_candidate():
    gate_script = Path("scripts/ci_regression_gate.py")
    result = subprocess.run(
        [sys.executable, str(gate_script), "--prod", "models/model_v1.json", "--cand", "models/model_v2.json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "REJECT" in result.stdout
