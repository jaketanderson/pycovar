"""Run the HCcH-TOCSY self-covariance in every mode of covar.py.

Writes one 4D per mode, 4D_HCCH_python_<mode>.pipe, for comparing the ways
the complex shared dimension can enter the covariance.
"""
import subprocess
import sys

MODES = ["real", "complex", "split"]

for mode in MODES:
    print(f"=== {mode} ===", flush=True)
    subprocess.run([sys.executable, "covar.py", mode], check=True)
