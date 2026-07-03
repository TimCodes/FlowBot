"""Rung 4a: ES-MCCFR blueprint for heads-up no-limit Hold'em.

Trains the same external-sampling MCCFR (hulhe_mccfr.ESMCCFRTrainer) on the
abstracted HUNL game (nlhe_engine: fold/call/half-pot/pot/all-in, 200 BB
stacks). This is the Libratus-recipe *blueprint* -- a coarse equilibrium
approximation for the whole game. The second half of the recipe, real-time
depth-limited subgame re-solving, refines play from the turn onward and is
the next milestone; the blueprint alone is what we benchmark first.

Usage:
    .venv\\Scripts\\python hunl_blueprint.py --iterations 30000
"""

from __future__ import annotations

import argparse
import pickle
import time

from card_abstraction import EquityBucketer
from hulhe_mccfr import (CallAgent, ESMCCFRTrainer, PolicyAgent, RandomAgent,
                         mbb, play_match)
from nlhe_engine import BIG_BLIND, NLHEState


def deal_nlhe(cards):
    return NLHEState((tuple(cards[0:2]), tuple(cards[2:4])), tuple(cards[4:9]))


def main():
    parser = argparse.ArgumentParser(
        description="External-sampling MCCFR blueprint for HUNL")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--buckets", type=int, default=8)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--eval-hands", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default="hunl_blueprint.pkl")
    args = parser.parse_args()

    bucketer = EquityBucketer(args.buckets, args.samples, args.seed)
    trainer = ESMCCFRTrainer(bucketer, args.seed, state_factory=deal_nlhe)

    print(f"ES-MCCFR blueprint on HUNL (200BB, f/c/h/p/a): "
          f"{args.iterations} iterations, {args.buckets} buckets, "
          f"{args.samples} MC samples", flush=True)
    start = time.perf_counter()
    for i in range(1, args.iterations + 1):
        trainer.iteration()
        if i % args.eval_every == 0 or i == args.iterations:
            policy = trainer.average_policy()
            agent = PolicyAgent(policy, bucketer, seed=i)
            vs_rand = play_match(agent, RandomAgent(seed=i), args.eval_hands,
                                 seed=i, state_factory=deal_nlhe)
            vs_call = play_match(agent, CallAgent(), args.eval_hands,
                                 seed=i + 1, state_factory=deal_nlhe)
            elapsed = time.perf_counter() - start
            print(f"iter {i:>7}: {len(trainer.nodes):>7} infosets, "
                  f"EHS cache {len(bucketer.cache):>7}, "
                  f"vs random {mbb(vs_rand, BIG_BLIND):>+8.1f} mbb/hand, "
                  f"vs call {mbb(vs_call, BIG_BLIND):>+8.1f} mbb/hand  "
                  f"({elapsed:.0f}s)", flush=True)

    with open(args.save, "wb") as f:
        pickle.dump({"policy": trainer.average_policy(),
                     "buckets": args.buckets,
                     "samples": args.samples,
                     "iterations": args.iterations}, f)
    print(f"Blueprint saved to {args.save}", flush=True)


if __name__ == "__main__":
    main()
