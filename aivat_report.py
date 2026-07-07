"""AIVAT-lite: unbiased luck-adjusted win-rate estimate from a match log.

Full AIVAT (Burch et al. 2018) needs value estimates at every decision and
chance node; that requires a trained value function. This is the honest
subset built from control variates whose expectations are exactly zero,
so the adjusted estimator stays unbiased:

  1. All-in board luck (beta = 1, in chips): when the hand ends with an
     all-in call before the river, both hole cards are revealed and the rest
     of the board is pure chance. The correction replaces the realized
     outcome with the all-in equity: luck = winnings - (equity*pot - stake).
     Conditional on the cards and the all-in, E[luck] = 0.
  2. Preflop hole-card luck (beta fitted by OLS): feature = our hole class's
     equity vs a random hand minus 0.5. Over random deals E[feature] = 0, so
     subtracting beta*feature is unbiased for ANY beta; OLS just picks the
     variance-minimizing one.

Expect a meaningful but not miraculous variance cut (roughly 1.5-3x fewer
hands for the same error bar); the full-AIVAT ~10x needs the value network.

Usage:
    .venv\\Scripts\\python aivat_report.py --log-jsonl slumbot_10k.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations

from treys import Card

from card_abstraction import preflop_class
from holdem_engine import BOARD_N, FULL_DECK, _evaluator
from nlhe_engine import BIG_BLIND, STACK
from slumbot_client import parse_action

POT_ALL_IN = 2 * STACK


def allin_equity(our_hole, their_hole, board_prefix, rng_samples=2000):
    """Our equity when all-in on `board_prefix`, both hands known.

    Enumerates remaining runouts exactly when feasible (flop: 990, turn: 44),
    otherwise Monte Carlo (preflop: C(48,5) is too many).
    """
    blocked = set(our_hole) | set(their_hole) | set(board_prefix)
    remaining = [c for c in FULL_DECK if c not in blocked]
    need = 5 - len(board_prefix)
    ours, theirs, prefix = list(our_hole), list(their_hole), list(board_prefix)

    def share(runout):
        board = prefix + list(runout)
        r0 = _evaluator.evaluate(ours, board)
        r1 = _evaluator.evaluate(theirs, board)
        return 1.0 if r0 < r1 else (0.5 if r0 == r1 else 0.0)

    if need == 0:
        return share([])
    if math.comb(len(remaining), need) <= 2000:
        runouts = list(combinations(remaining, need))
        return sum(share(r) for r in runouts) / len(runouts)
    import random
    rng = random.Random(7)
    total = 0.0
    for _ in range(rng_samples):
        total += share(rng.sample(remaining, need))
    return total / rng_samples


def allin_luck_chips(record) -> float:
    """Chips of pure board luck in an all-in hand (0 for other hands)."""
    parsed = parse_action(record["action"])
    st = parsed.get("allin_call_st")
    if st is None or not record.get("bot_hole_cards"):
        return 0.0
    ours = [Card.new(c) for c in record["hole_cards"]]
    theirs = [Card.new(c) for c in record["bot_hole_cards"]]
    board_prefix = [Card.new(c) for c in record["board"][:BOARD_N[st]]]
    equity = allin_equity(ours, theirs, board_prefix)
    expected = equity * POT_ALL_IN - STACK
    return record["winnings"] - expected


def preflop_luck_feature(record, class_equity_cache) -> float:
    """Hole-card quality feature with known zero mean over random deals."""
    hole = [Card.new(c) for c in record["hole_cards"]]
    cls = preflop_class(hole)
    if cls not in class_equity_cache:
        from card_abstraction import EquityBucketer
        b = EquityBucketer(samples=2000, seed=11)
        class_equity_cache[cls] = b.hand_strength(hole, ())
    return class_equity_cache[cls] - 0.5


def summarize(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n), math.sqrt(var)


def main():
    parser = argparse.ArgumentParser(description="Luck-adjusted match report")
    parser.add_argument("--log-jsonl", required=True)
    args = parser.parse_args()

    records = []
    with open(args.log_jsonl) as f:
        for line in f:
            records.append(json.loads(line))
    n = len(records)
    print(f"{n} hands in {args.log_jsonl}")

    y = [r["winnings"] for r in records]
    cache: dict[str, float] = {}
    z = [yi - allin_luck_chips(r) for yi, r in zip(y, records)]
    x = [preflop_luck_feature(r, cache) for r in records]

    # OLS beta of z on x (E[x] = 0 by construction).
    x_mean = sum(x) / n
    z_mean = sum(z) / n
    denom = sum((xi - x_mean) ** 2 for xi in x)
    beta = sum((xi - x_mean) * (zi - z_mean) for xi, zi in zip(x, z)) / denom
    adjusted = [zi - beta * xi for zi, xi in zip(z, x)]

    to_mbb = 1000 / BIG_BLIND
    for label, vals in (("raw", y), ("all-in adjusted", z),
                        ("AIVAT-lite (all-in + preflop OLS)", adjusted)):
        mean, se, sd = summarize(vals)
        print(f"{label:>36}: {mean * to_mbb:>+8.1f} ± {se * to_mbb:.1f} "
              f"mbb/hand   (per-hand sd {sd * to_mbb:.0f} mbb)")
    _, se_raw, _ = summarize(y)
    _, se_adj, _ = summarize(adjusted)
    print(f"\nVariance reduction: {(se_raw / se_adj) ** 2:.2f}x "
          f"(equivalent to {(se_raw / se_adj) ** 2:.1f}x more hands); "
          f"OLS beta = {beta:.0f} chips per unit preflop equity")


if __name__ == "__main__":
    main()
