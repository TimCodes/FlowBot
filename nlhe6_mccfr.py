"""Linear external-sampling MCCFR for 6-max no-limit Hold'em (rung 5.3).

Multiplayer deltas from the heads-up trainer (hulhe_mccfr.ESMCCFRTrainer),
following the Pluribus blueprint recipe (Brown & Sandholm, Science 2019 --
see research_6max.md):

  * Every seat takes a turn as the traverser each iteration; terminal values
    are the traverser's own net chips (there is no zero-sum shortcut with 6
    players). Once the traverser has folded, its payoff is fixed at
    -contrib[traverser], so the traversal short-circuits.
  * Linear CFR: regret and average-strategy contributions on iteration t are
    weighted by t, washing out early noise quadratically faster. Regret
    matching normalizes, so the growing weights cost nothing.
  * Negative-regret pruning: after `prune_after` iterations, actions whose
    cumulative regret is below `prune_threshold` are skipped in `prune_prob`
    of iterations (the rest traverse everything so mistakes can recover).
    Pruning only engages at nodes with some positive regret -- at all-negative
    nodes regret matching plays uniformly, and skipping there would bias the
    node value. Regrets are floored at `regret_floor` so recovery stays
    possible.
  * Infoset keys are seat-prefixed: with fixed blinds, seat number *is*
    position (UTG opens a different range than the button).

With 3+ players CFR keeps no Nash guarantee (it still removes iteratively
strictly dominated strategies); the success metric is mbb/hand against agent
pools via `play_table`, which rotates the hero through all six seats.

Usage:
    .venv\\Scripts\\python nlhe6_blueprint.py --iterations 30000
"""

from __future__ import annotations

import random

from card_abstraction import EquityBucketer
from holdem_engine import FULL_DECK
from hulhe_mccfr import CallAgent, Node, RandomAgent, mbb, regret_matching
from nlhe6_engine import BIG_BLIND, NLHE6State, NUM_PLAYERS

__all__ = ["deal_nlhe6", "infoset_key6", "Linear6MCCFRTrainer", "PolicyAgent6",
           "RandomAgent", "CallAgent", "play_table", "mbb", "BIG_BLIND"]


def deal_nlhe6(cards, num_players: int = NUM_PLAYERS) -> NLHE6State:
    """State factory: 2n+5 sampled cards -> a fresh 6-max hand."""
    holes = tuple(tuple(cards[2 * i:2 * i + 2]) for i in range(num_players))
    return NLHE6State(holes, tuple(cards[2 * num_players:2 * num_players + 5]))


def infoset_key6(state: NLHE6State, bucketer: EquityBucketer) -> str:
    hole = state.holes[state.to_act]
    label = bucketer.label(hole, state.board_revealed(), state.street)
    return f"{state.to_act}|{label}|{state.history_str()}"


class Linear6MCCFRTrainer:
    def __init__(self, bucketer: EquityBucketer, seed: int = 0,
                 num_players: int = NUM_PLAYERS, state_factory=deal_nlhe6,
                 prune_threshold: float = -3.0e8, prune_prob: float = 0.95,
                 prune_after: int = 20_000, regret_floor: float = -3.1e8):
        self.bucketer = bucketer
        self.num_players = num_players
        self.state_factory = state_factory
        self.nodes: dict[str, Node] = {}
        self.rng = random.Random(seed)
        self.t = 0
        self.prune_threshold = prune_threshold
        self.prune_prob = prune_prob
        self.prune_after = prune_after
        self.regret_floor = regret_floor

    def iteration(self):
        """One Linear-MCCFR iteration: every seat traverses a fresh deal."""
        self.t += 1
        prune = (self.t > self.prune_after
                 and self.rng.random() < self.prune_prob)
        deck_need = 2 * self.num_players + 5
        for seat in range(self.num_players):
            cards = self.rng.sample(FULL_DECK, deck_need)
            self._traverse(self.state_factory(cards, self.num_players),
                           seat, prune)

    def _traverse(self, state: NLHE6State, traverser: int,
                  prune: bool) -> float:
        if state.is_terminal():
            return state.utility(traverser)
        if state.folded[traverser]:
            return -state.contrib[traverser]  # fixed no matter the runout

        actions = state.legal_actions()
        key = infoset_key6(state, self.bucketer)
        node = self.nodes.get(key)
        if node is None:
            node = self.nodes[key] = Node(actions)
        sigma = regret_matching(node.regret_sum)

        if state.to_act == traverser:
            can_prune = prune and any(r > 0 for r in node.regret_sum)
            utils = {}
            value = 0.0
            for i, a in enumerate(actions):
                if can_prune and node.regret_sum[i] < self.prune_threshold:
                    continue
                u = self._traverse(state.apply(a), traverser, prune)
                utils[i] = u
                value += sigma[i] * u
            for i, u in utils.items():
                node.regret_sum[i] = max(
                    self.regret_floor,
                    node.regret_sum[i] + self.t * (u - value))
            return value

        # Another seat: accumulate its average strategy, sample one action.
        for i in range(len(actions)):
            node.strategy_sum[i] += self.t * sigma[i]
        action = self.rng.choices(actions, weights=sigma)[0]
        return self._traverse(state.apply(action), traverser, prune)

    def average_policy(self) -> dict[str, list[float]]:
        policy = {}
        for key, node in self.nodes.items():
            total = sum(node.strategy_sum)
            if total > 0:
                policy[key] = [s / total for s in node.strategy_sum]
            else:
                policy[key] = [1.0 / len(node.actions)] * len(node.actions)
        return policy


class PolicyAgent6:
    """Plays a trained average policy; uniform over legal actions if unseen."""

    def __init__(self, policy, bucketer, seed=0):
        self.policy = policy
        self.bucketer = bucketer
        self.rng = random.Random(seed)

    def act(self, state):
        actions = state.legal_actions()
        probs = self.policy.get(infoset_key6(state, self.bucketer))
        if probs is None or len(probs) != len(actions):
            return self.rng.choice(actions)
        return self.rng.choices(actions, weights=probs)[0]


def play_table(hero, villains, hands: int, seed: int = 0,
               num_players: int = NUM_PLAYERS,
               state_factory=deal_nlhe6, per_hand: bool = False):
    """Full-table match: hero rotates through all seats vs n-1 villains.

    Returns the hero's average chips/hand (or the per-hand list when
    `per_hand` is set, for standard-error reporting). Seat rotation
    balances the large positional edge the way seat alternation does for HU.
    """
    assert len(villains) == num_players - 1
    rng = random.Random(seed)
    results = []
    for h in range(hands):
        hero_seat = h % num_players
        seats = (list(villains[:hero_seat]) + [hero]
                 + list(villains[hero_seat:]))
        state = state_factory(rng.sample(FULL_DECK, 2 * num_players + 5),
                              num_players)
        while not state.is_terminal():
            state = state.apply(seats[state.to_act].act(state))
        results.append(state.utility(hero_seat))
    if per_hand:
        return results
    return sum(results) / hands
