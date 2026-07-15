"""Measure shadow-vs-real pot drift at river decisions.

Every re-solve variant computes chip-exact EVs from the SHADOW state, whose
contributions come from applying abstract actions -- but the real bets were
whatever Slumbot and we actually wagered. If the shadow pot systematically
differs from the real pot, the resolver prices calls/bets against the wrong
stakes (an inflated shadow pot makes real calls look cheap -> overcalls),
while the blueprint's coarse bucket policy is robust to the same drift.

Usage:
    .venv\\Scripts\\python diag_pot_drift.py --log-jsonl slumbot_2500_deepstack.jsonl
"""

from __future__ import annotations

import argparse
import json

from nlhe_engine import ACTION_PROFILES
from diag_safe import first_river_decision_prefix
from slumbot_client import parse_action, replay_abstract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-jsonl", required=True)
    ap.add_argument("--actions", default="ext")
    ap.add_argument("--sample", type=int, default=300)
    args = ap.parse_args()
    state_cls = ACTION_PROFILES[args.actions]

    drifts, rows = [], []
    for line in open(args.log_jsonl):
        if len(drifts) >= args.sample:
            break
        rec = json.loads(line)
        if not rec.get("action") or len(rec.get("board") or []) < 5:
            continue
        prefix = first_river_decision_prefix(rec["action"], rec["client_pos"])
        if prefix is None:
            continue
        parsed = parse_action(prefix)
        if "error" in parsed:
            continue
        real_pot = parsed["total_contrib"][0] + parsed["total_contrib"][1]
        shadow = replay_abstract(prefix, rec["hole_cards"], rec["board"],
                                 state_cls=state_cls)
        if shadow is None or shadow.is_terminal():
            continue
        shadow_pot = sum(shadow.contrib)
        drifts.append(shadow_pot / real_pot)
        if len(rows) < 8:
            rows.append((rec["hand"], real_pot, shadow_pot, prefix))

    drifts.sort()
    n = len(drifts)
    print(f"{n} river decisions ({args.actions} profile)")
    print(f"shadow/real pot ratio: median={drifts[n // 2]:.2f}  "
          f"p10={drifts[n // 10]:.2f}  p90={drifts[9 * n // 10]:.2f}  "
          f"min={drifts[0]:.2f}  max={drifts[-1]:.2f}")
    over = sum(1 for d in drifts if d > 1.25) / n
    under = sum(1 for d in drifts if d < 0.8) / n
    print(f"inflated >1.25x: {over:.0%}   deflated <0.8x: {under:.0%}")
    print("\nexamples (hand, real pot, shadow pot):")
    for h, rp, sp, pref in rows:
        print(f"  hand {h}: real={rp:6d} shadow={sp:6d}  action={pref!r}")


if __name__ == "__main__":
    main()
