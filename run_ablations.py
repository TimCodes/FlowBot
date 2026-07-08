"""Per-upgrade ablation study vs Slumbot (rung 4 follow-up).

The combined-upgrade test came out flat after luck adjustment, so this
driver measures each upgrade in isolation, 10k hands each, sequentially
(one Slumbot session at a time):

    hs2_only       E[HS^2] blueprint, naive translation, no re-solving
    harmonic_only  baseline blueprint + pseudo-harmonic translation
    resolve_only   baseline blueprint + river re-solving

Each match logs to ablation_<name>.jsonl; matches are resumable (rerunning
this driver skips completed hands), and an AIVAT-lite report is printed
after each match. Compare everything against the two anchors already
measured: baseline -230 +/- 158, all-upgrades -214 +/- 143 (luck-adjusted).

Usage:
    .venv\\Scripts\\python run_ablations.py [--hands 10000]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

ABLATIONS = [
    ("hs2_only", ["--blueprint", "hunl_blueprint_hs2.pkl"]),
    ("harmonic_only", ["--blueprint", "hunl_blueprint_big.pkl",
                       "--translation", "harmonic"]),
    ("resolve_only", ["--blueprint", "hunl_blueprint_big.pkl", "--resolve"]),
]


def main():
    parser = argparse.ArgumentParser(description="Sequential ablation matches")
    parser.add_argument("--hands", type=int, default=10000)
    args = parser.parse_args()

    for name, extra in ABLATIONS:
        log = f"ablation_{name}.jsonl"
        print(f"\n=== ablation {name}: {args.hands} hands -> {log} ===",
              flush=True)
        start = time.perf_counter()
        subprocess.run([sys.executable, "slumbot_client.py",
                        "--hands", str(args.hands), "--log-jsonl", log,
                        *extra], check=True)
        print(f"=== {name} done in {(time.perf_counter() - start) / 3600:.1f}h; "
              f"luck-adjusted report: ===", flush=True)
        subprocess.run([sys.executable, "aivat_report.py",
                        "--log-jsonl", log], check=True)
    print("\nAll ablations complete.", flush=True)


if __name__ == "__main__":
    main()
