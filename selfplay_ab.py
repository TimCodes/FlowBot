"""In-engine A/B: DeepStack-resolved river play vs pure blueprint.

The decisive control experiment for the re-solving investigation. Inside our
own engine the resolver's model of the opponent is correct BY CONSTRUCTION
(the opponent literally plays the blueprint the CBVs are computed against).

  * Resolver wins here + loses vs Slumbot  -> implementation sound; the live
    failures are model mismatch (Slumbot's conditional holdings differ from
    any model we can build without observing it), and no re-solve variant
    can help head-to-head without an empirical opponent model.
  * Resolver loses here too                -> implementation defect after all.

Usage:
    .venv\\Scripts\\python selfplay_ab.py --hands 1000 --resolve-iters 5000
"""

from __future__ import annotations

import argparse
import pickle
import random
import time

from card_abstraction import EquityBucketer
from deepstack_resolver import DeepStackResolver
from hulhe_mccfr import PolicyAgent, mbb
from nlhe_engine import ACTION_PROFILES, BIG_BLIND
from holdem_engine import FULL_DECK
from river_resolver import opponent_range


class ResolvedRiverAgent(PolicyAgent):
    """Plays the blueprint until the river, then DeepStack-style re-solves.

    In self-play the shadow state IS the real state, so the trace needed for
    our-range estimation is reconstructed from the state's history string
    (every action in-engine is already abstract).
    """

    def __init__(self, policy, bucketer, seed=0, iterations=5000,
                 cbv_mode="br"):
        super().__init__(policy, bucketer, seed=seed)
        self.resolver = DeepStackResolver(iterations=iterations, seed=seed,
                                          cbv_mode=cbv_mode)

    def act(self, state):
        if state.street != 3:
            return super().act(state)
        try:
            trace = self._trace_from_history(state)
            our_range = opponent_range(self.policy, self.bucketer, trace,
                                       state.to_act, (),
                                       state.board_revealed())
            dist = self.resolver.resolve_deepstack(
                state, state.holes[state.to_act], our_range,
                self.policy, self.bucketer)
            actions = list(dist)
            return self.rng.choices(actions,
                                    weights=[dist[a] for a in actions])[0]
        except Exception:
            return super().act(state)

    def _trace_from_history(self, state):
        """Replay the abstract history to recover per-decision records."""
        replay = type(state)(state.holes, state.board)
        trace = []
        for street_hist in state.history_str().split("/"):
            for a in street_hist:
                trace.append((replay.to_act, replay.street,
                              replay.history_str(),
                              tuple(replay.legal_actions()), a))
                replay = replay.apply(a)
        return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blueprint", default="hunl_blueprint_ext.pkl")
    ap.add_argument("--hands", type=int, default=1000)
    ap.add_argument("--resolve-iters", type=int, default=5000)
    ap.add_argument("--cbv-mode", choices=("br", "blueprint"), default="br")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    saved = pickle.load(open(args.blueprint, "rb"))
    bucketer = EquityBucketer(saved["buckets"], saved["samples"], args.seed,
                              mode=saved.get("mode", "ehs"))
    state_cls = ACTION_PROFILES[saved.get("actions", "std")]
    resolved = ResolvedRiverAgent(saved["policy"], bucketer,
                                  seed=args.seed,
                                  iterations=args.resolve_iters,
                                  cbv_mode=args.cbv_mode)
    vanilla = PolicyAgent(saved["policy"], bucketer, seed=args.seed + 1)

    rng = random.Random(args.seed)
    total, river_total, river_n = 0, 0, 0
    start = time.perf_counter()
    for h in range(1, args.hands + 1):
        a_seat = h % 2
        cards9 = rng.sample(FULL_DECK, 9)
        state = state_cls((tuple(cards9[0:2]), tuple(cards9[2:4])),
                          tuple(cards9[4:9]))
        seats = (resolved, vanilla) if a_seat == 0 else (vanilla, resolved)
        touched_river = False
        while not state.is_terminal():
            if state.street == 3 and seats[state.to_act] is resolved:
                touched_river = True
            state = state.apply(seats[state.to_act].act(state))
        payoff = state.utility(a_seat)
        total += payoff
        if touched_river:
            river_total += payoff
            river_n += 1
        if h % 200 == 0 or h == args.hands:
            print(f"hand {h:>5}: resolved-agent {mbb(total / h, BIG_BLIND):+8.1f} "
                  f"mbb/hand overall; river-decision hands "
                  f"{mbb(river_total / max(river_n, 1), BIG_BLIND):+8.1f} "
                  f"(n={river_n})  [{time.perf_counter() - start:.0f}s]",
                  flush=True)


if __name__ == "__main__":
    main()
