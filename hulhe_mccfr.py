"""External-sampling MCCFR for heads-up limit Texas Hold'em (rung 3).

Full HULHE has ~3x10^14 decision points -- Cepheus needed 262 TB of regret
tables to solve it losslessly. Here the game is abstracted (169 preflop
classes + equity buckets postflop, see card_abstraction.py), shrinking the
infoset space to buckets x betting-sequences, and trained with
external-sampling MCCFR (Lanctot et al. 2009): the traversing player explores
all actions; the opponent and chance are sampled.

Exact exploitability is no longer computable at this scale, so evaluation is
head-to-head: mbb/hand (milli-big-blinds per hand) against baseline agents
over seat-alternated matches. Slumbot-style benchmarking (rung 4) is the same
metric against a stronger opponent.

Usage:
    .venv\\Scripts\\python hulhe_mccfr.py --iterations 8000
    .venv\\Scripts\\python hulhe_mccfr.py --iterations 8000 --buckets 12 --samples 100
"""

from __future__ import annotations

import argparse
import pickle
import random
import time

from card_abstraction import EquityBucketer
from holdem_engine import BIG_BLIND, CALL, FULL_DECK, HULHEState


def deal_hulhe(cards):
    """Default state factory: 9 sampled cards -> a fresh HULHE hand."""
    return HULHEState((tuple(cards[0:2]), tuple(cards[2:4])), tuple(cards[4:9]))


def infoset_key(state, bucketer: EquityBucketer) -> str:
    hole = state.holes[state.to_act]
    label = bucketer.label(hole, state.board_revealed(), state.street)
    return f"{label}|{state.history_str()}"


def regret_matching(regrets):
    positive = [max(r, 0.0) for r in regrets]
    total = sum(positive)
    if total > 0:
        return [p / total for p in positive]
    return [1.0 / len(regrets)] * len(regrets)


class Node:
    __slots__ = ("regret_sum", "strategy_sum", "actions")

    def __init__(self, actions):
        self.actions = actions
        self.regret_sum = [0.0] * len(actions)
        self.strategy_sum = [0.0] * len(actions)


class ESMCCFRTrainer:
    def __init__(self, bucketer: EquityBucketer, seed: int = 0,
                 state_factory=deal_hulhe):
        self.bucketer = bucketer
        self.nodes: dict[str, Node] = {}
        self.rng = random.Random(seed)
        self.state_factory = state_factory

    def iteration(self):
        """One MCCFR iteration: a fresh deal and traversal per update player."""
        for update_player in (0, 1):
            state = self.state_factory(self.rng.sample(FULL_DECK, 9))
            self._traverse(state, update_player)

    def _traverse(self, state: HULHEState, update_player: int) -> float:
        if state.is_terminal():
            return state.utility(update_player)

        actions = state.legal_actions()
        key = infoset_key(state, self.bucketer)
        node = self.nodes.get(key)
        if node is None:
            node = self.nodes[key] = Node(actions)
        sigma = regret_matching(node.regret_sum)

        if state.to_act == update_player:
            utils = [self._traverse(state.apply(a), update_player)
                     for a in actions]
            value = sum(s * u for s, u in zip(sigma, utils))
            for i in range(len(actions)):
                node.regret_sum[i] += utils[i] - value
            return value

        # Opponent node: accumulate their average strategy, sample one action.
        for i in range(len(actions)):
            node.strategy_sum[i] += sigma[i]
        action = self.rng.choices(actions, weights=sigma)[0]
        return self._traverse(state.apply(action), update_player)

    def average_policy(self) -> dict[str, list[float]]:
        policy = {}
        for key, node in self.nodes.items():
            total = sum(node.strategy_sum)
            if total > 0:
                policy[key] = [s / total for s in node.strategy_sum]
            else:
                policy[key] = [1.0 / len(node.actions)] * len(node.actions)
        return policy


# --- Agents and match play -------------------------------------------------

class PolicyAgent:
    """Plays a trained average policy; uniform over legal actions if unseen."""

    def __init__(self, policy, bucketer, seed=0):
        self.policy = policy
        self.bucketer = bucketer
        self.rng = random.Random(seed)

    def act(self, state):
        actions = state.legal_actions()
        probs = self.policy.get(infoset_key(state, self.bucketer))
        if probs is None or len(probs) != len(actions):
            return self.rng.choice(actions)
        return self.rng.choices(actions, weights=probs)[0]


class RandomAgent:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, state):
        return self.rng.choice(state.legal_actions())


class CallAgent:
    """Call-station: always checks/calls, never folds, never raises."""

    def act(self, state):
        return CALL


def play_match(agent_a, agent_b, hands: int, seed: int = 0,
               state_factory=deal_hulhe) -> float:
    """Seat-alternated match; returns agent_a's average chips/hand."""
    rng = random.Random(seed)
    total = 0
    for h in range(hands):
        a_seat = h % 2
        state = state_factory(rng.sample(FULL_DECK, 9))
        seats = (agent_a, agent_b) if a_seat == 0 else (agent_b, agent_a)
        while not state.is_terminal():
            state = state.apply(seats[state.to_act].act(state))
        total += state.utility(a_seat)
    return total / hands


def mbb(chips_per_hand: float, big_blind: int = BIG_BLIND) -> float:
    return chips_per_hand / big_blind * 1000


def main():
    parser = argparse.ArgumentParser(
        description="External-sampling MCCFR on abstracted HULHE")
    parser.add_argument("--iterations", type=int, default=8000)
    parser.add_argument("--buckets", type=int, default=8)
    parser.add_argument("--samples", type=int, default=50,
                        help="Monte Carlo rollouts per hand-strength estimate")
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--eval-hands", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default="hulhe_policy.pkl")
    args = parser.parse_args()

    bucketer = EquityBucketer(args.buckets, args.samples, args.seed)
    trainer = ESMCCFRTrainer(bucketer, args.seed)

    print(f"ES-MCCFR on HULHE: {args.iterations} iterations, "
          f"{args.buckets} buckets, {args.samples} MC samples", flush=True)
    start = time.perf_counter()
    for i in range(1, args.iterations + 1):
        trainer.iteration()
        if i % args.eval_every == 0 or i == args.iterations:
            policy = trainer.average_policy()
            agent = PolicyAgent(policy, bucketer, seed=i)
            vs_rand = play_match(agent, RandomAgent(seed=i), args.eval_hands, seed=i)
            vs_call = play_match(agent, CallAgent(), args.eval_hands, seed=i + 1)
            elapsed = time.perf_counter() - start
            print(f"iter {i:>7}: {len(trainer.nodes):>6} infosets, "
                  f"EHS cache {len(bucketer.cache):>7}, "
                  f"vs random {mbb(vs_rand):>+8.1f} mbb/hand, "
                  f"vs call {mbb(vs_call):>+8.1f} mbb/hand  ({elapsed:.0f}s)",
                  flush=True)

    with open(args.save, "wb") as f:
        pickle.dump({"policy": trainer.average_policy(),
                     "buckets": args.buckets,
                     "samples": args.samples,
                     "iterations": args.iterations}, f)
    print(f"Policy saved to {args.save}", flush=True)


if __name__ == "__main__":
    main()
