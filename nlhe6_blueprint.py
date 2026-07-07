"""Rung 5.3: Linear-MCCFR blueprint for 6-max no-limit Hold'em.

Trains nlhe6_mccfr.Linear6MCCFRTrainer on the abstracted 6-max game
(nlhe6_engine: fold/call/half-pot/pot/all-in, 200 BB stacks, side pots),
with equity buckets rolled out against 5 random opponents. This is the
Pluribus-recipe blueprint; the second half of the recipe -- real-time
depth-limited search over four biased continuation strategies -- is the
next milestone (research_6max.md, step 5.5).

Evaluation is a full-table pool match (no Slumbot exists for 6-max): the
blueprint rotates through all six seats against five random agents and
against five call-stations. With 3+ players mbb/hand vs pools *is* the
success metric; exploitability is neither guaranteed nor computable.

Usage:
    .venv\\Scripts\\python nlhe6_blueprint.py --iterations 30000
    .venv\\Scripts\\python nlhe6_blueprint.py --resume --iterations 2000000
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

from card_abstraction import EquityBucketer
from hunl_blueprint import keep_system_awake
from nlhe6_mccfr import (BIG_BLIND, CallAgent, Linear6MCCFRTrainer,
                         PolicyAgent6, RandomAgent, mbb, play_table)
from nlhe6_engine import NUM_PLAYERS


def main():
    parser = argparse.ArgumentParser(
        description="Linear ES-MCCFR blueprint for 6-max NLHE")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--buckets", type=int, default=8)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--opponents", type=int, default=NUM_PLAYERS - 1,
                        help="random opponents in equity rollouts")
    parser.add_argument("--mode", choices=("ehs", "ehs2"), default="ehs",
                        help="bucket feature: mean equity or RMS E[HS^2]")
    parser.add_argument("--prune-after", type=int, default=20000,
                        help="iterations before negative-regret pruning")
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--eval-hands", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default="nlhe6_blueprint.pkl")
    parser.add_argument("--state-file", default="nlhe6_trainer_state.pkl",
                        help="full trainer state (regrets) for --resume")
    parser.add_argument("--resume", action="store_true",
                        help="continue from --state-file if it exists")
    parser.add_argument("--allow-sleep", action="store_true",
                        help="do not hold the system awake while training")
    args = parser.parse_args()

    if not args.allow_sleep:
        keep_system_awake()

    bucketer = EquityBucketer(args.buckets, args.samples, args.seed,
                              num_opponents=args.opponents, mode=args.mode)
    trainer = Linear6MCCFRTrainer(bucketer, args.seed,
                                  prune_after=args.prune_after)

    start_iter = 0
    if args.resume and os.path.exists(args.state_file):
        with open(args.state_file, "rb") as f:
            state = pickle.load(f)
        trainer.nodes = state["nodes"]
        trainer.t = start_iter = state["iteration"]
        # The RNG restarts from the seed; the post-resume sample sequence
        # differs from an uninterrupted run. Harmless for MCCFR.
        print(f"Resumed from {args.state_file}: iteration {start_iter:,}, "
              f"{len(trainer.nodes):,} infosets", flush=True)

    def atomic_dump(obj, path):
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    def save_checkpoint(policy, iterations_done):
        atomic_dump({"policy": policy,
                     "buckets": args.buckets,
                     "samples": args.samples,
                     "opponents": args.opponents,
                     "mode": args.mode,
                     "iterations": iterations_done}, args.save)
        atomic_dump({"nodes": trainer.nodes,
                     "iteration": iterations_done}, args.state_file)

    print(f"Linear ES-MCCFR blueprint on 6-max NLHE (200BB, f/c/h/p/a): "
          f"{args.iterations} iterations, {args.buckets} buckets "
          f"({args.mode}, vs {args.opponents} opps), {args.samples} MC "
          f"samples, pruning after {args.prune_after}", flush=True)
    start = time.perf_counter()
    for i in range(start_iter + 1, args.iterations + 1):
        trainer.iteration()
        if i % args.eval_every == 0 or i == args.iterations:
            policy = trainer.average_policy()
            agent = PolicyAgent6(policy, bucketer, seed=i)
            vs_rand = play_table(
                agent, [RandomAgent(seed=i + j) for j in range(5)],
                args.eval_hands, seed=i)
            vs_call = play_table(
                agent, [CallAgent() for _ in range(5)],
                args.eval_hands, seed=i + 1)
            save_checkpoint(policy, i)
            elapsed = time.perf_counter() - start
            print(f"iter {i:>7}: {len(trainer.nodes):>8} infosets, "
                  f"EHS cache {len(bucketer.cache):>8}, "
                  f"vs random {mbb(vs_rand, BIG_BLIND):>+8.1f} mbb/hand, "
                  f"vs call {mbb(vs_call, BIG_BLIND):>+8.1f} mbb/hand  "
                  f"({elapsed:.0f}s, checkpoint saved)", flush=True)

    print(f"Blueprint saved to {args.save}", flush=True)


if __name__ == "__main__":
    main()
