"""Chance-sampling Counterfactual Regret Minimization (CFR) for Kuhn poker.

Step 1 of the poker-bot research ladder: implement vanilla CFR (and the CFR+
variant) on the smallest nontrivial poker game, and verify convergence against
the known analytical Nash equilibrium.

Kuhn poker
----------
3-card deck (J=1, Q=2, K=3). Each player antes 1 chip and receives one card.
Player 0 acts first. Actions: pass ('p') or bet 1 chip ('b').
Terminal histories: pp, bp, bb, pbp, pbb.

Known equilibrium properties (one-parameter family, alpha in [0, 1/3]):
  * Game value to player 0 is -1/18.
  * P0 opens J (bluff) with prob alpha, Q never, K with prob 3*alpha.
  * P0 facing a bet after checking: folds J, calls Q with alpha + 1/3, calls K.
  * P1 facing a bet: folds J, calls Q with 1/3, always calls K.
  * P1 after a check: bets J with 1/3, checks Q, always bets K.

No third-party dependencies; pure standard library.
"""

from __future__ import annotations

import random

PASS, BET = 0, 1
ACTIONS = ("p", "b")
CARDS = (1, 2, 3)
CARD_NAMES = {1: "J", 2: "Q", 3: "K"}
DEALS = [(a, b) for a in CARDS for b in CARDS if a != b]  # 6 deals, 1/6 each

TERMINALS = ("pp", "bp", "bb", "pbp", "pbb")


def is_terminal(history: str) -> bool:
    return history in TERMINALS


def payoff_to_current(cards, history: str) -> int:
    """Terminal payoff to the player whose turn it *would* be (len(history) % 2)."""
    player = len(history) % 2
    if history in ("bp", "pbp"):  # opponent folded to a bet
        return 1
    win = cards[player] > cards[1 - player]
    amount = 1 if history == "pp" else 2  # showdown after check-check or bet-call
    return amount if win else -amount


class Node:
    """Regret and average-strategy accumulators for one information set."""

    __slots__ = ("regret_sum", "strategy_sum")

    def __init__(self):
        self.regret_sum = [0.0, 0.0]
        self.strategy_sum = [0.0, 0.0]

    def current_strategy(self):
        positive = [max(r, 0.0) for r in self.regret_sum]
        total = sum(positive)
        if total > 0:
            return [p / total for p in positive]
        return [0.5, 0.5]

    def average_strategy(self):
        total = sum(self.strategy_sum)
        if total > 0:
            return [s / total for s in self.strategy_sum]
        return [0.5, 0.5]


class KuhnCFRTrainer:
    """Chance-sampling CFR. Set plus=True for CFR+ (regret flooring +
    linearly weighted strategy averaging)."""

    def __init__(self, plus: bool = False, seed: int | None = None):
        self.nodes: dict[str, Node] = {}
        self.plus = plus
        self.rng = random.Random(seed)
        self.iteration = 0

    def train(self, iterations: int) -> float:
        """Run CFR iterations; returns the average sampled game value for P0."""
        total = 0.0
        cards = list(CARDS)
        for _ in range(iterations):
            self.iteration += 1
            self.rng.shuffle(cards)
            total += self._cfr(cards, "", 1.0, 1.0)
        return total / iterations

    def _cfr(self, cards, history: str, p0: float, p1: float) -> float:
        """Returns expected utility for the player to act at this node."""
        if is_terminal(history):
            return payoff_to_current(cards, history)

        player = len(history) % 2
        info_set = str(cards[player]) + history
        node = self.nodes.setdefault(info_set, Node())

        strategy = node.current_strategy()
        my_reach = p0 if player == 0 else p1
        # CFR+ uses linearly increasing averaging weights (t * reach).
        weight = my_reach * (self.iteration if self.plus else 1.0)
        for a in range(2):
            node.strategy_sum[a] += weight * strategy[a]

        util = [0.0, 0.0]
        node_util = 0.0
        for a in range(2):
            next_history = history + ACTIONS[a]
            if player == 0:
                util[a] = -self._cfr(cards, next_history, p0 * strategy[a], p1)
            else:
                util[a] = -self._cfr(cards, next_history, p0, p1 * strategy[a])
            node_util += strategy[a] * util[a]

        opp_reach = p1 if player == 0 else p0
        for a in range(2):
            node.regret_sum[a] += opp_reach * (util[a] - node_util)
            if self.plus:
                node.regret_sum[a] = max(node.regret_sum[a], 0.0)  # regret flooring

        return node_util

    def average_strategies(self) -> dict[str, list[float]]:
        return {k: n.average_strategy() for k, n in self.nodes.items()}


def best_response_value(avg: dict[str, list[float]], br_player: int) -> float:
    """Expected value of the best response to `avg` for `br_player`.

    Public-tree traversal carrying a belief vector over deals. At best-responder
    nodes, deals are grouped by the responder's card (their information set) and
    the max-value action is chosen per group.
    """

    def strat(info_set):
        return avg.get(info_set, [0.5, 0.5])

    def walk(history, reach):  # reach: {deal: prob incl. chance and opponent strategy}
        player = len(history) % 2
        if is_terminal(history):
            total = 0.0
            for deal, p in reach.items():
                u = payoff_to_current(deal, history)
                total += p * (u if player == br_player else -u)
            return total
        if player == br_player:
            total = 0.0
            for card in {d[br_player] for d in reach}:
                sub = {d: p for d, p in reach.items() if d[br_player] == card}
                total += max(walk(history + a, sub) for a in ACTIONS)
            return total
        total = 0.0
        for ai, a in enumerate(ACTIONS):
            sub = {d: p * strat(str(d[player]) + history)[ai] for d, p in reach.items()}
            total += walk(history + a, sub)
        return total

    return walk("", {d: 1.0 / len(DEALS) for d in DEALS})


def nash_conv(avg: dict[str, list[float]]) -> float:
    """NashConv = sum of best-response values; 0 iff `avg` is a Nash equilibrium."""
    return best_response_value(avg, 0) + best_response_value(avg, 1)


def describe(info_set: str) -> str:
    card, history = CARD_NAMES[int(info_set[0])], info_set[1:]
    player = len(history) % 2
    return f"P{player} {card} '{history or '-'}'"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train CFR on Kuhn poker")
    parser.add_argument("--iterations", type=int, default=200_000)
    parser.add_argument("--plus", action="store_true", help="use CFR+ variant")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    trainer = KuhnCFRTrainer(plus=args.plus, seed=args.seed)
    value = trainer.train(args.iterations)
    avg = trainer.average_strategies()

    variant = "CFR+" if args.plus else "vanilla CFR"
    print(f"{variant}, {args.iterations:,} iterations")
    print(f"Sampled game value for P0: {value:+.4f}   (analytical: {-1/18:+.4f})")
    print(f"NashConv (exploitability): {nash_conv(avg):.5f}\n")
    print(f"{'infoset':<14} {'pass/fold':>10} {'bet/call':>10}")
    for key in sorted(avg, key=lambda k: (len(k), k)):
        s = avg[key]
        print(f"{describe(key):<14} {s[PASS]:>10.3f} {s[BET]:>10.3f}")


if __name__ == "__main__":
    main()
