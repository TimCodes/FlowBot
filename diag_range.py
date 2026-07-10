"""Diagnostic: is the resolver's opponent-range estimate collapsing?

Replays logged river hands offline (no API) and, at each of our river
decisions, measures how opponent_range behaves under the given blueprint:

  * uniform_fallback: the observed opponent line had total weight 0 under
    the blueprint (an "impossible line"), so opponent_range returns a
    uniform range over ALL combos -- the resolver then solves vs a range
    that bears no relation to the real opponent.
  * support / entropy: size and concentration of the returned range.

Hypothesis under test: the sharper ext blueprint (E[HS^2], 6 actions, 5M
iters) assigns exact-zero probability to more lines, so the uniform
fallback fires far more often than with the softer std blueprint --
which would make ext+resolve solve against garbage ranges.

Usage:
    .venv\\Scripts\\python diag_range.py --blueprint hunl_blueprint_ext.pkl \\
        --log-jsonl slumbot_10k_ext_resolve.jsonl --sample 400
"""

from __future__ import annotations

import argparse
import json
import math
import pickle

from treys import Card

from card_abstraction import EquityBucketer
from nlhe_engine import ACTION_PROFILES
from river_resolver import opponent_range
from slumbot_client import parse_action, replay_abstract


def range_entropy_bits(rng):
    return -sum(p * math.log2(p) for p in rng.values() if p > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blueprint", required=True)
    ap.add_argument("--log-jsonl", required=True)
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--smooth", type=float, default=0.0)
    args = ap.parse_args()

    saved = pickle.load(open(args.blueprint, "rb"))
    bucketer = EquityBucketer(saved["buckets"], saved["samples"], 0,
                              mode=saved.get("mode", "ehs"))
    policy = saved["policy"]
    # Must replay with the blueprint's OWN action profile, or the trace's
    # legal sets won't match the stored policy vectors (an earlier version
    # of this script omitted this and produced a spurious 97% "collapse").
    state_cls = ACTION_PROFILES[saved.get("actions", "std")]

    n_decisions = 0
    n_uniform_fallback = 0
    supports, entropies = [], []
    hands_seen = 0

    with open(args.log_jsonl) as f:
        for line in f:
            if hands_seen >= args.sample:
                break
            rec = json.loads(line)
            action = rec.get("action")
            if not action or not rec.get("board") or len(rec["board"]) < 5:
                continue  # never reached the river
            hands_seen += 1
            # Reconstruct our river decision points from the full action string
            # by parsing prefixes: whenever it's our turn on the river.
            trace = []
            shadow = replay_abstract(action, rec["hole_cards"], rec["board"],
                                     trace_out=trace, state_cls=state_cls)
            if shadow is None:
                continue
            our_seat = rec["client_pos"]
            board = tuple(Card.new(c) for c in rec["board"])
            our_hole = tuple(Card.new(c) for c in rec["hole_cards"])
            rng = opponent_range(policy, bucketer, trace, 1 - our_seat,
                                 our_hole, board, smooth=args.smooth)
            n_decisions += 1
            support = len(rng)
            supports.append(support)
            entropies.append(range_entropy_bits(rng))
            # C(45,2)=990 unblocked combos with a 5-card board; the uniform
            # fallback returns exactly that many equal-weight entries.
            if support >= 990 and max(rng.values()) - min(rng.values()) < 1e-12:
                n_uniform_fallback += 1

    if not n_decisions:
        print("no river decisions found in sample")
        return
    print(f"blueprint: {args.blueprint} (mode={saved.get('mode','ehs')}, "
          f"actions={saved.get('actions','std')})")
    print(f"river hands sampled:      {hands_seen}")
    print(f"range estimates:          {n_decisions}")
    print(f"uniform-fallback rate:    {n_uniform_fallback / n_decisions:.1%}  "
          f"({n_uniform_fallback}/{n_decisions})")
    print(f"mean range support:       {sum(supports) / len(supports):.0f} combos")
    print(f"mean range entropy:       {sum(entropies) / len(entropies):.1f} bits "
          f"(uniform-990 = {math.log2(990):.1f})")


if __name__ == "__main__":
    main()
