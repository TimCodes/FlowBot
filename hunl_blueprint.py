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
import os
import pickle
import sys
import time


def keep_system_awake():
    """Stop Windows from sleeping while training runs.

    Uses SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED), which
    is scoped to this process: the OS clears it automatically on exit, so no
    power settings are permanently changed. The display may still sleep.
    """
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

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
    parser.add_argument("--state-file", default="hunl_trainer_state.pkl",
                        help="full trainer state (regrets) for --resume")
    parser.add_argument("--resume", action="store_true",
                        help="continue from --state-file if it exists")
    parser.add_argument("--allow-sleep", action="store_true",
                        help="do not hold the system awake while training")
    args = parser.parse_args()

    if not args.allow_sleep:
        keep_system_awake()

    bucketer = EquityBucketer(args.buckets, args.samples, args.seed)
    trainer = ESMCCFRTrainer(bucketer, args.seed, state_factory=deal_nlhe)

    start_iter = 0
    if args.resume and os.path.exists(args.state_file):
        with open(args.state_file, "rb") as f:
            state = pickle.load(f)
        trainer.nodes = state["nodes"]
        start_iter = state["iteration"]
        # Note: the RNG restarts from the seed, so the post-resume sample
        # sequence differs from an uninterrupted run. Harmless for MCCFR.
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
                     "iterations": iterations_done}, args.save)
        atomic_dump({"nodes": trainer.nodes,
                     "iteration": iterations_done}, args.state_file)

    print(f"ES-MCCFR blueprint on HUNL (200BB, f/c/h/p/a): "
          f"{args.iterations} iterations, {args.buckets} buckets, "
          f"{args.samples} MC samples", flush=True)
    start = time.perf_counter()
    for i in range(start_iter + 1, args.iterations + 1):
        trainer.iteration()
        if i % args.eval_every == 0 or i == args.iterations:
            policy = trainer.average_policy()
            agent = PolicyAgent(policy, bucketer, seed=i)
            vs_rand = play_match(agent, RandomAgent(seed=i), args.eval_hands,
                                 seed=i, state_factory=deal_nlhe)
            vs_call = play_match(agent, CallAgent(), args.eval_hands,
                                 seed=i + 1, state_factory=deal_nlhe)
            save_checkpoint(policy, i)
            elapsed = time.perf_counter() - start
            print(f"iter {i:>7}: {len(trainer.nodes):>7} infosets, "
                  f"EHS cache {len(bucketer.cache):>7}, "
                  f"vs random {mbb(vs_rand, BIG_BLIND):>+8.1f} mbb/hand, "
                  f"vs call {mbb(vs_call, BIG_BLIND):>+8.1f} mbb/hand  "
                  f"({elapsed:.0f}s, checkpoint saved)", flush=True)

    print(f"Blueprint saved to {args.save}", flush=True)


if __name__ == "__main__":
    main()
