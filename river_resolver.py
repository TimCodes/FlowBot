"""Depth-limited subgame re-solving from the turn or river (rung 4).

At a turn or river decision the blueprint is replaced by a fresh solve of
the remaining subgame:

  1. Opponent range: every hole combo consistent with our cards and the
     visible board, weighted by the product of the blueprint's probabilities
     for the abstract actions the opponent actually took (Bayesian reach).
     This is *unsafe* re-solving -- it trusts the blueprint's model of the
     opponent; safe re-solving (gadget games, CFR-D) is the follow-up.
  2. Subgame: the shadow abstract state from slumbot_client with the pot,
     stacks, and betting history as they stand. From the river only betting
     remains; from the turn the river card is a chance node, sampled per
     CFR iteration, with our post-river infosets keyed by the actual card.
  3. Solve with external-sampling CFR (opponent hole -- and river card when
     starting from the turn -- sampled each iteration), then act from the
     average strategy at the root.

Opponent infosets are clustered into strength-percentile buckets so their
strategy converges in few iterations: exact treys rank on complete boards,
mean rank over a shared sample of rivers on the turn. Our own hand is never
bucketed -- showdowns use exact evaluations.
"""

from __future__ import annotations

import random

from holdem_engine import BOARD_N, FULL_DECK, _evaluator
from hulhe_mccfr import regret_matching
from nlhe_engine import DOUBLE_POT

OPP_BUCKETS = 10
TURN_SCORE_RIVERS = 10  # shared river sample for turn-strength ranking


def opponent_range(policy, bucketer, trace, opp_seat, our_hole, board):
    """Blueprint-implied distribution over opponent hole-card combos.

    trace: decision records from replay_abstract(trace_out=...).
    `board` should be the *revealed* cards only. Returns {(c1, c2): weight}
    normalized to sum 1; combos the blueprint would never have played this
    way get weight 0 and are dropped.
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


class SubgameResolver:
    def __init__(self, iterations: int = 2000, seed: int = 0,
                 from_street: int = 3, cap_pot: bool = False):
        assert from_street in (2, 3)
        self.iterations = iterations
        self.from_street = from_street
        # cap_pot: forbid our own DOUBLE_POT overbets in the subgame. Unsafe
        # re-solving trusts the blueprint's opponent model, so a large bet
        # against an out-of-model opponent (e.g. Slumbot) is dangerous; the
        # capstone run showed ext+resolve regressing -777 mbb/hand from
        # exactly this. Opponent actions keep full legality (they are modeled).
        self.cap_pot = cap_pot
        self.rng = random.Random(seed)

    def resolve(self, shadow, our_hole, opp_range) -> dict[str, float]:
        """Solve the subgame rooted at `shadow` (our turn to act, street 2-3).

        Returns {action: probability} for shadow.legal_actions().
        """
        street0 = shadow.street
        assert street0 in (2, 3), "resolver supports turn and river roots"
        our_seat = shadow.to_act
        opp_seat = 1 - our_seat
        board4 = list(shadow.board[:4])
        known = set(our_hole) | set(shadow.board[:BOARD_N[street0]])
        deck_left = [c for c in FULL_DECK if c not in known]

        holes = list(opp_range.keys())
        probs = [opp_range[h] for h in holes]

        def build_river_map(riv):
            """Rank-percentile buckets for opponent hands given river card."""
            legal = [h for h in holes if riv not in h]
            board = board4 + [riv]
            ranked = sorted(legal,
                            key=lambda h: _evaluator.evaluate(list(h), board))
            return {h: min(OPP_BUCKETS - 1, i * OPP_BUCKETS // len(ranked))
                    for i, h in enumerate(ranked)}

        river_maps: dict[int, dict] = {}
        turn_map = None
        if street0 == 2:
            # Turn strength: mean exact rank over a shared river sample.
            sample_rivs = self.rng.sample(deck_left,
                                          min(TURN_SCORE_RIVERS,
                                              len(deck_left)))

            def turn_score(h):
                rivers = [r for r in sample_rivs if r not in h]
                return sum(_evaluator.evaluate(list(h), board4 + [r])
                           for r in rivers) / len(rivers)

            ranked = sorted(holes, key=turn_score)
            turn_map = {h: min(OPP_BUCKETS - 1, i * OPP_BUCKETS // len(ranked))
                        for i, h in enumerate(ranked)}

        def opp_bucket(state, opp_hole):
            if state.street >= 3:
                riv = state.board[4]
                bucket_map = river_maps.get(riv)
                if bucket_map is None:
                    bucket_map = river_maps[riv] = build_river_map(riv)
                return bucket_map.get(opp_hole, 0)
            return turn_map[opp_hole]

        nodes: dict[str, list] = {}  # key -> [regret_sum, strategy_sum, acts]

        def node_for(key, actions):
            n = nodes.get(key)
            if n is None:
                n = nodes[key] = [[0.0] * len(actions),
                                  [0.0] * len(actions), actions]
            return n

        def infoset_key(state, opp_hole):
            riv = state.board[4] if state.street >= 3 else ""
            if state.to_act == our_seat:
                return f"u{state.street}|{riv}|{state.history_str()}"
            return (f"o{state.street}|{riv}|{opp_bucket(state, opp_hole)}|"
                    f"{state.history_str()}")

        def traverse(state, opp_hole, update_player):
            if state.is_terminal():
                return state.utility(update_player)
            actions = state.legal_actions()
            if self.cap_pot and state.to_act == our_seat:
                actions = [a for a in actions if a != DOUBLE_POT]
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

        def make_state(opp_hole, riv):
            s = shadow._clone()
            s.holes = ((our_hole, opp_hole) if our_seat == 0
                       else (opp_hole, our_hole))
            if riv is not None:
                s.board = tuple(board4) + (riv,)
            return s

        for _ in range(self.iterations):
            opp_hole = self.rng.choices(holes, weights=probs)[0]
            riv = None
            if street0 == 2:
                riv = self.rng.choice(
                    [c for c in deck_left if c not in opp_hole])
            for update_player in (our_seat, opp_seat):
                traverse(make_state(opp_hole, riv), opp_hole, update_player)

        root_riv = shadow.board[4] if street0 == 3 else ""
        root_key = f"u{street0}|{root_riv}|{shadow.history_str()}"
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


# Backwards-compatible name (river-only default).
RiverResolver = SubgameResolver
