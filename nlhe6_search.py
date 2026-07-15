"""Rung 5.5: real-time depth-limited search for 6-max NLHE (Pluribus-style).

The second half of the Pluribus recipe (Brown & Sandholm, Science 2019; Brown,
Sandholm & Amos, NeurIPS 2018 -- see research_6max.md section 3.2 and 6.3).
The blueprint plays the first betting round; on every later decision the agent
re-solves the subgame rooted at the current public state:

  * Opponent ranges: for each live opponent, every hole-card candidate is
    weighted by the product of the blueprint probabilities of the actions
    that opponent actually took (reach weights), replayed from the public
    history. Candidates use only public information -- legality of actions
    never depends on hidden cards.
  * Determinization: each solver iteration samples opponent holes from those
    ranges and re-samples the unrevealed board from the unseen deck (using
    the true future board would leak information).
  * Depth limit: the subgame is solved to the end of the current street
    (exactly to showdown when the root is the river). At each leaf, every
    live player picks one of four continuation strategies -- the blueprint,
    or the blueprint biased toward folding, calling, or raising (the biased
    action's probability multiplied by 5, renormalized). The choice is a
    regret-minimized meta-decision inside the solve, which is what makes
    depth-limited solving sound-ish where a scalar leaf value is not.
  * The solve runs external-sampling MCCFR with every live seat taking a
    turn as traverser; the hero's move is the average strategy at the root
    infoset.

Deliberate simplification, documented for the writeup: the hero's own hand is
fixed at its actual holding ("unsafe" search) rather than solved across its
full range as Pluribus does for unpredictability. Range-balanced search is
the upgrade path; against agent pools the unsafe version measures the same
thing (does search beat the raw blueprint?).

Usage (evaluate search agent vs a table of blueprint agents):
    .venv\\Scripts\\python nlhe6_search.py --blueprint nlhe6_blueprint.pkl \\
        --hands 200 --search-iters 200
"""

from __future__ import annotations

import argparse
import pickle
import random
import time

from card_abstraction import EquityBucketer
from holdem_engine import FULL_DECK
from hulhe_mccfr import Node, regret_matching, mbb
from nlhe6_engine import BIG_BLIND, CALL, FOLD, NLHE6State, STACK
from nlhe6_mccfr import PolicyAgent6, deal_nlhe6, infoset_key6, play_table

CONTINUATIONS = (None, "fold", "call", "raise")  # k = 4, as in Pluribus
BIAS_FACTOR = 5.0


def biased_probs(probs, actions, bias, factor=BIAS_FACTOR):
    """Pluribus leaf continuation: multiply the biased action class's
    probability by `factor` and renormalize. bias None = the blueprint."""
    if bias is None:
        return probs
    w = list(probs)
    for i, a in enumerate(actions):
        if (bias == "fold" and a == FOLD) or (bias == "call" and a == CALL) \
                or (bias == "raise" and a not in (FOLD, CALL)):
            w[i] *= factor
    total = sum(w)
    if total <= 0:
        return [1.0 / len(w)] * len(w)
    return [x / total for x in w]


class Blueprint:
    """Average-policy lookup with a uniform fallback for unseen infosets."""

    def __init__(self, policy, bucketer):
        self.policy = policy
        self.bucketer = bucketer

    def probs(self, state):
        actions = state.legal_actions()
        p = self.policy.get(infoset_key6(state, self.bucketer))
        if p is None or len(p) != len(actions):
            return [1.0 / len(actions)] * len(actions)
        return p


def replay_decisions(state):
    """Reconstruct (state_before, action) for every decision in the public
    history. Legal actions never depend on hidden cards, so replaying with
    the state's own hole tuple leaks nothing."""
    fresh = NLHE6State(state.holes, state.board)
    nodes = []
    for hist in state.hists:
        for ch in hist:
            nodes.append((fresh, ch))
            fresh = fresh.apply(ch)
    return nodes


def opponent_ranges(state, blueprint, hero):
    """Reach-weighted hole-pair range for each live opponent.

    Weight(hole) = product over the opponent's observed actions of the
    blueprint probability of that action given `hole`. Ranges are kept
    independent per opponent (joint card-removal is handled at sampling
    time); hero's cards and the revealed board are excluded.
    """
    revealed = set(state.board_revealed())
    blocked = revealed | set(state.holes[hero])
    deck = [c for c in FULL_DECK if c not in blocked]
    pairs = [(deck[i], deck[j]) for i in range(len(deck))
             for j in range(i + 1, len(deck))]
    decisions = replay_decisions(state)
    ranges = {}
    for seat in range(state.n):
        if seat == hero or state.folded[seat]:
            continue
        seat_nodes = [(st, ch) for st, ch in decisions if st.to_act == seat]
        weights = []
        for hole in pairs:
            w = 1.0
            for st, ch in seat_nodes:
                actions = st.legal_actions()
                label = blueprint.bucketer.label(
                    hole, st.board_revealed(), st.street)
                key = f"{seat}|{label}|{st.history_str()}"
                p = blueprint.policy.get(key)
                if p is None or len(p) != len(actions):
                    w *= 1.0 / len(actions)
                else:
                    w *= p[actions.index(ch)]
                if w == 0.0:
                    break
            weights.append(w)
        if sum(weights) <= 0.0:
            weights = [1.0] * len(pairs)
        ranges[seat] = (pairs, weights)
    return ranges


class SubgameSolver:
    """Depth-limited ES-MCCFR re-solve of the subgame rooted at `state`.

    `warm` warm-starts every subgame node at the blueprint (Brown's
    warm-starting idea, scaled to the root pot): regret mass proportional
    to the blueprint probabilities makes the *simulated opponents* open at
    blueprint play instead of uniform noise, and the anchored strategy
    average makes the hero degrade gracefully to the blueprint when the
    iteration budget is too small to learn a deviation. Without this, a
    small-budget solve models opponents as random and its answer loses to
    the very blueprint it started from.
    """

    def __init__(self, state, blueprint, hero, seed=0, warm=20.0):
        self.root = state
        self.blueprint = blueprint
        self.hero = hero
        self.root_street = state.street
        self.rng = random.Random(seed)
        self.nodes: dict[str, Node] = {}
        self.ranges = opponent_ranges(state, blueprint, hero)
        self.live = [i for i in range(state.n) if not state.folded[i]]
        self.warm = warm
        self.pot = sum(state.contrib)

    def _make_node(self, state, actions):
        node = Node(actions)
        if self.warm > 0:
            probs = self.blueprint.probs(state)
            node.regret_sum = [self.warm * self.pot * p for p in probs]
            node.strategy_sum = [self.warm * p for p in probs]
        return node

    # -- determinization ----------------------------------------------------

    def _sample_state(self) -> NLHE6State:
        """Clone the root with opponent holes drawn from their reach-weighted
        ranges and the unrevealed board re-dealt from the unseen deck."""
        used = set(self.root.holes[self.hero]) | set(self.root.board_revealed())
        holes = list(self.root.holes)
        for seat, (pairs, weights) in self.ranges.items():
            for _ in range(64):  # rejection-sample around card collisions
                hole = self.rng.choices(pairs, weights=weights)[0]
                if used.isdisjoint(hole):
                    break
            else:
                free = [c for c in FULL_DECK if c not in used]
                hole = tuple(self.rng.sample(free, 2))
            holes[seat] = hole
            used |= set(hole)
        free = [c for c in FULL_DECK if c not in used]
        revealed = list(self.root.board_revealed())
        board = revealed + self.rng.sample(free, 5 - len(revealed))
        det = self.root._clone()
        det.holes = tuple(holes)
        det.board = tuple(board)
        return det

    # -- solving ------------------------------------------------------------

    def solve(self, iterations: int):
        for _ in range(iterations):
            for traverser in self.live:
                self._traverse(self._sample_state(), traverser)

    def root_strategy(self):
        """Average strategy at the hero's root infoset (accumulated while
        other seats traversed); falls back to positive-regret matching."""
        key = infoset_key6(self.root, self.blueprint.bucketer)
        node = self.nodes.get(key)
        actions = self.root.legal_actions()
        if node is None:
            return actions, [1.0 / len(actions)] * len(actions)
        total = sum(node.strategy_sum)
        if total > 0:
            return actions, [s / total for s in node.strategy_sum]
        return actions, regret_matching(node.regret_sum)

    def _traverse(self, state, traverser):
        if state.is_terminal():
            return state.utility(traverser)
        if state.folded[traverser]:
            return -state.contrib[traverser]
        if state.street > self.root_street:
            return self._leaf(state, traverser)

        actions = state.legal_actions()
        key = infoset_key6(state, self.blueprint.bucketer)
        node = self.nodes.get(key)
        if node is None:
            node = self.nodes[key] = self._make_node(state, actions)
        sigma = regret_matching(node.regret_sum)

        if state.to_act == traverser:
            utils = [self._traverse(state.apply(a), traverser)
                     for a in actions]
            value = sum(s * u for s, u in zip(sigma, utils))
            for i in range(len(actions)):
                node.regret_sum[i] += utils[i] - value
            return value

        for i in range(len(actions)):
            node.strategy_sum[i] += sigma[i]
        action = self.rng.choices(actions, weights=sigma)[0]
        return self._traverse(state.apply(action), traverser)

    # -- depth-limit leaves -------------------------------------------------

    def _leaf(self, state, traverser):
        """Each live, non-all-in player picks one of four continuation
        strategies (a regret-minimized meta-decision); the hand is then
        rolled out under the chosen profile."""
        choosers = [i for i in range(state.n)
                    if not state.folded[i] and state.contrib[i] < STACK]
        cont = {}
        for seat in choosers:
            if seat == traverser:
                continue
            node = self._meta_node(state, seat)
            sigma = regret_matching(node.regret_sum)
            for i in range(len(CONTINUATIONS)):
                node.strategy_sum[i] += sigma[i]
            cont[seat] = self.rng.choices(
                range(len(CONTINUATIONS)), weights=sigma)[0]

        if traverser not in choosers:
            return self._rollout(state, cont, traverser)

        node = self._meta_node(state, traverser)
        sigma = regret_matching(node.regret_sum)
        utils = []
        for c in range(len(CONTINUATIONS)):
            cont[traverser] = c
            utils.append(self._rollout(state, cont, traverser))
        value = sum(s * u for s, u in zip(sigma, utils))
        for i in range(len(CONTINUATIONS)):
            node.regret_sum[i] += utils[i] - value
        return value

    def _meta_node(self, state, seat):
        label = self.blueprint.bucketer.label(
            state.holes[seat], state.board_revealed(), state.street)
        key = f"cont|{seat}|{label}|{state.history_str()}"
        node = self.nodes.get(key)
        if node is None:
            node = self.nodes[key] = Node(list(CONTINUATIONS))
            if self.warm > 0:  # anchor the meta-choice at "play blueprint"
                node.regret_sum[0] = self.warm * self.pot
                node.strategy_sum[0] = self.warm
        return node

    def _rollout(self, state, cont, traverser):
        while not state.is_terminal():
            actions = state.legal_actions()
            probs = self.blueprint.probs(state)
            bias = CONTINUATIONS[cont.get(state.to_act, 0)]
            w = biased_probs(probs, actions, bias)
            state = state.apply(self.rng.choices(actions, weights=w)[0])
        return state.utility(traverser)


class SearchAgent6:
    """Blueprint preflop; depth-limited subgame re-solving postflop."""

    def __init__(self, policy, bucketer, search_iters=200, seed=0, warm=20.0):
        self.blueprint = Blueprint(policy, bucketer)
        self.fallback = PolicyAgent6(policy, bucketer, seed=seed)
        self.search_iters = search_iters
        self.warm = warm
        self.rng = random.Random(seed)

    def act(self, state):
        if state.street == 0:
            return self.fallback.act(state)  # Pluribus: blueprint preflop
        solver = SubgameSolver(state, self.blueprint, hero=state.to_act,
                               seed=self.rng.randrange(1 << 30),
                               warm=self.warm)
        solver.solve(self.search_iters)
        actions, probs = solver.root_strategy()
        return self.rng.choices(actions, weights=probs)[0]


def main():
    parser = argparse.ArgumentParser(
        description="Depth-limited search agent vs a blueprint table")
    parser.add_argument("--blueprint", default="nlhe6_blueprint.pkl")
    parser.add_argument("--hands", type=int, default=200)
    parser.add_argument("--search-iters", type=int, default=200)
    parser.add_argument("--warm", type=float, default=20.0,
                        help="blueprint warm-start mass (0 disables)")
    parser.add_argument("--paired", action="store_true",
                        help="difference against a blueprint hero on the "
                             "same deals (deal luck cancels)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.blueprint, "rb") as f:
        bp = pickle.load(f)
    bucketer = EquityBucketer(
        bp["buckets"], bp["samples"], args.seed,
        num_opponents=bp.get("opponents", 5), mode=bp.get("mode", "ehs"))
    policy = bp["policy"]
    print(f"Loaded {args.blueprint}: {len(policy):,} infosets, "
          f"{bp['buckets']} buckets, {bp['iterations']:,} training iters",
          flush=True)

    def table(hero):
        # Fresh villains with identical seeds each run, so paired runs see
        # the same deals (play_table draws once per hand) and the same
        # villain rng streams until trajectories diverge.
        villains = [PolicyAgent6(policy, bucketer, seed=args.seed + 1 + j)
                    for j in range(5)]
        return play_table(hero, villains, args.hands, seed=args.seed,
                          per_hand=True)

    start = time.perf_counter()
    results = table(SearchAgent6(policy, bucketer, args.search_iters,
                                 seed=args.seed, warm=args.warm))
    label = f"search({args.search_iters}, warm={args.warm})"
    if args.paired:
        base = table(PolicyAgent6(policy, bucketer, seed=args.seed + 99))
        results = [s - b for s, b in zip(results, base)]
        label += " minus paired blueprint hero"
    elapsed = time.perf_counter() - start
    n = len(results)
    mean = sum(results) / n
    var = sum((r - mean) ** 2 for r in results) / (n - 1)
    se = (var / n) ** 0.5
    print(f"{label} vs 5x blueprint over {n} hands: "
          f"{mbb(mean, BIG_BLIND):+.1f} ± {mbb(se, BIG_BLIND):.1f} "
          f"mbb/hand (1 SE) ({elapsed:.0f}s, {elapsed / n:.2f}s/hand)",
          flush=True)


if __name__ == "__main__":
    main()
