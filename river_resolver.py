"""Depth-limited subgame re-solving on the river (rung 4, Libratus recipe).

At each river decision the blueprint is replaced by a fresh, exact solve of
the remaining subgame:

  1. Opponent range: every hole combo consistent with our cards and the
     board (<= 990), weighted by the product of the blueprint's probabilities
     for the abstract actions the opponent actually took (Bayesian reach).
     This is *unsafe* re-solving -- it trusts the blueprint's model of the
     opponent; safe re-solving (gadget games, CFR-D) is the follow-up.
  2. Subgame: the shadow abstract state from slumbot_client with the pot,
     stacks, and betting history as they stand; only river betting remains,
     so the tree is tiny and showdowns are exact treys evaluations (no
     bucketing error on our side; the opponent's strategy is clustered into
     rank-percentile buckets so it converges in few iterations).
  3. Solve with external-sampling CFR (opponent hole sampled from the range
     each iteration), then act from the average strategy at the root.

Because the pot and history are real (not bucketed), the resolver repairs
the two main blueprint weaknesses at once: card abstraction (we use our
exact hand) and stale ranges (we condition on the actual line taken).
"""

from __future__ import annotations

import random

from holdem_engine import BOARD_N, FULL_DECK, _evaluator
from hulhe_mccfr import regret_matching

OPP_BUCKETS = 10


def opponent_range(policy, bucketer, trace, opp_seat, our_hole, board):
    """Blueprint-implied distribution over opponent hole-card combos.

    trace: decision records from replay_abstract(trace_out=...).
    Returns {(c1, c2): weight} normalized to sum 1; combos the blueprint
    would never have played this way get weight 0 and are dropped.
    """
    blocked = set(our_hole) | set(board)
    deck = [c for c in FULL_DECK if c not in blocked]
    opp_decisions = [t for t in trace if t[0] == opp_seat]
    weights = {}
    for i in range(len(deck)):
        for j in range(i + 1, len(deck)):
            hole = (deck[i], deck[j])
            w = 1.0
            for _, street, hist, legal, chosen in opp_decisions:
                label = bucketer.label(hole, board[:BOARD_N[street]], street)
                probs = policy.get(f"{label}|{hist}")
                if probs is None or len(probs) != len(legal):
                    w *= 1.0 / len(legal)
                else:
                    w *= probs[legal.index(chosen)]
                if w == 0.0:
                    break
            if w > 0.0:
                weights[hole] = w
    total = sum(weights.values())
    if total <= 0.0:  # blueprint says "impossible line": fall back to uniform
        combos = [(deck[i], deck[j]) for i in range(len(deck))
                  for j in range(i + 1, len(deck))]
        return {h: 1.0 / len(combos) for h in combos}
    return {h: w / total for h, w in weights.items()}


class RiverResolver:
    def __init__(self, iterations: int = 2000, seed: int = 0):
        self.iterations = iterations
        self.rng = random.Random(seed)

    def resolve(self, shadow, our_hole, opp_range) -> dict[str, float]:
        """Solve the river subgame rooted at `shadow` (our turn to act).

        Returns {action: probability} for shadow.legal_actions().
        """
        our_seat = shadow.to_act
        opp_seat = 1 - our_seat
        board = shadow.board

        # Cluster opponent hands into river-strength percentile buckets for
        # their infosets (exact hands are still used at showdown).
        holes = list(opp_range.keys())
        probs = [opp_range[h] for h in holes]
        ranked = sorted(holes,
                        key=lambda h: _evaluator.evaluate(list(h), list(board)))
        bucket_of = {h: min(OPP_BUCKETS - 1, i * OPP_BUCKETS // len(ranked))
                     for i, h in enumerate(ranked)}

        nodes: dict[str, list] = {}  # key -> [regret_sum, strategy_sum, acts]

        def node_for(key, actions):
            n = nodes.get(key)
            if n is None:
                n = nodes[key] = [[0.0] * len(actions),
                                  [0.0] * len(actions), actions]
            return n

        def infoset_key(state, opp_hole):
            if state.to_act == our_seat:
                return "u|" + state.history_str()
            return f"o{bucket_of[opp_hole]}|" + state.history_str()

        def traverse(state, opp_hole, update_player):
            if state.is_terminal():
                return state.utility(update_player)
            actions = state.legal_actions()
            regret, strat, _ = node_for(infoset_key(state, opp_hole), actions)
            sigma = regret_matching(regret)
            if state.to_act == update_player:
                utils = [traverse(state.apply(a), opp_hole, update_player)
                         for a in actions]
                value = sum(s * u for s, u in zip(sigma, utils))
                for k in range(len(actions)):
                    regret[k] += utils[k] - value
                return value
            for k in range(len(actions)):
                strat[k] += sigma[k]
            action = self.rng.choices(actions, weights=sigma)[0]
            return traverse(state.apply(action), opp_hole, update_player)

        def with_holes(opp_hole):
            s = shadow._clone()
            s.holes = ((our_hole, opp_hole) if our_seat == 0
                       else (opp_hole, our_hole))
            return s

        for _ in range(self.iterations):
            opp_hole = self.rng.choices(holes, weights=probs)[0]
            for update_player in (our_seat, opp_seat):
                traverse(with_holes(opp_hole), opp_hole, update_player)

        root_key = "u|" + shadow.history_str()
        actions = shadow.legal_actions()
        node = nodes.get(root_key)
        if node is None:
            return {a: 1.0 / len(actions) for a in actions}
        # The root is OUR node: its strategy_sum only accumulates during the
        # opponent's traversals, which is exactly the average strategy.
        total = sum(node[1])
        if total <= 0.0:
            return {a: 1.0 / len(actions) for a in actions}
        return {a: node[1][k] / total for k, a in enumerate(node[2])}
