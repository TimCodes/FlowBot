"""Safe (gadget-game) river re-solving -- CFR-D style, Burch et al. 2014.

Why: unsafe re-solving best-responds to the blueprint's *model* of the
opponent's range, so when that model is wrong the re-solved strategy can be
far worse than the blueprint (measured here: the ext blueprint models big
bets as bluff-heavy; the unsafe re-solve therefore overcalled and lost
-1721 mbb/hand on river-decision hands vs Slumbot's value-heavy reality).

The safe construction:

  1. OUR range at the subgame root is estimated the same way as the
     opponent's (blueprint reach over the observed line), because the
     guarantee is about our *range* strategy, not our actual hand.
  2. For each opponent hand h, estimate CBV(h): the value h could get by
     best-responding to our BLUEPRINT strategy in this subgame (a bucketed
     best-response traversal).
  3. Solve a gadget game: at the root the opponent, knowing h, chooses
     Terminate (payoff CBV(h) -- "your new strategy can't hurt me, I take
     the blueprint value") or Follow (enter the subgame). CFR on the gadget
     yields a strategy for our whole range such that no opponent hand can
     do meaningfully better than its blueprint value -- our re-solve can
     refine but not add exploitability (up to estimation error).
  4. At the table we play our actual hand's bucket of that range strategy.

Approximations vs the paper (all noted honestly): CBVs come from a bucketed
best response at decision time rather than exact values stored during the
blueprint solve; ranges are blueprint-reach estimates; card-removal between
our range and the opponent's is handled by rejection at sampling time but
ignored inside the bucketed CBV showdown mix. River-only (no chance nodes).
"""

from __future__ import annotations

import random
from bisect import bisect_left, bisect_right

from holdem_engine import _evaluator
from hulhe_mccfr import regret_matching

CBV_GROUPS = 30      # opponent rank-percentile groups for CBV estimation
TERMINATE, FOLLOW = "T", "F"


def _rank(hole, board):
    return _evaluator.evaluate(list(hole), list(board))


class _OurBuckets:
    """Our range grouped by blueprint policy label ("3:k" river buckets).

    Weights, per-bucket sorted member ranks (for P(win) lookups), and the
    blueprint mix pi(bucket, hist) drive the CBV best-response traversal.
    """

    def __init__(self, our_range, board, bucketer, policy):
        self.policy = policy
        self.by_label: dict[str, dict] = {}
        for g, w in our_range.items():
            label = bucketer.label(g, board, 3)
            b = self.by_label.setdefault(label, {"weight": 0.0, "ranks": []})
            b["weight"] += w
            b["ranks"].append(_rank(g, board))
        for b in self.by_label.values():
            b["ranks"].sort()
        self.labels = list(self.by_label)

    def pi(self, label, hist, legal):
        probs = self.policy.get(f"{label}|{hist}")
        if probs is None or len(probs) != len(legal):
            return [1.0 / len(legal)] * len(legal)
        return probs

    def showdown_value(self, rank_h, weights, chips):
        """Opp's expected chips at showdown vs the weighted bucket mix."""
        total_w = sum(weights.values())
        if total_w <= 0:
            return 0.0
        value = 0.0
        for label, w in weights.items():
            ranks = self.by_label[label]["ranks"]
            n = len(ranks)
            # treys: lower rank wins. Opp's h beats members with larger rank.
            beats = n - bisect_right(ranks, rank_h)
            ties = bisect_right(ranks, rank_h) - bisect_left(ranks, rank_h)
            value += w * chips * (beats - (n - beats - ties)) / n
        return value / total_w


def compute_cbvs(shadow, opp_range, our_buckets, opp_seat, mode="br",
                 policy=None, bucketer=None):
    """CBV per opponent hand against our blueprint range.

    mode="br": the opponent best-responds (upper values -- but against a
    weak blueprint these are enormously loose, and a gadget constrained by
    loose values is free to wander far from good head-to-head play).
    mode="blueprint": the opponent continues per its own blueprint policy
    (self-play continuation values -- tight, the DeepStack-style constraint;
    requires `policy` and `bucketer`).

    Opponent hands are grouped into CBV_GROUPS rank percentiles; the group's
    median hand stands in at showdowns and policy lookups. Returns
    ({hole: cbv}, {hole: group}).
    """
    board = shadow.board
    holes = sorted(opp_range, key=lambda h: _rank(h, board))
    groups = {}
    for i, h in enumerate(holes):
        groups[h] = min(CBV_GROUPS - 1, i * CBV_GROUPS // len(holes))
    rep_rank, rep_label = {}, {}
    for gidx in set(groups.values()):
        members = [h for h in holes if groups[h] == gidx]
        rep = members[len(members) // 2]
        rep_rank[gidx] = _rank(rep, board)
        if mode == "blueprint":
            rep_label[gidx] = bucketer.label(rep, board, 3)

    def opp_pi(gidx, hist, legal):
        probs = policy.get(f"{rep_label[gidx]}|{hist}") if policy else None
        if probs is None or len(probs) != len(legal):
            return [1.0 / len(legal)] * len(legal)
        return probs

    def br(state, gidx, weights):
        if state.is_terminal():
            if state.folded is not None:
                if state.folded == opp_seat:
                    return -state.contrib[opp_seat]
                return state.contrib[1 - opp_seat]
            return our_buckets.showdown_value(rep_rank[gidx], weights,
                                              state.contrib[0])
        legal = state.legal_actions()
        if state.to_act == opp_seat:
            children = [br(state.apply(a), gidx, weights) for a in legal]
            if mode == "br":  # opponent best-responds (loose upper values)
                return max(children)
            # mode == "blueprint": opponent continues per its own blueprint
            # (tight self-play continuation values -- what DeepStack's carried
            # trunk values approximate).
            pi = opp_pi(gidx, state.history_str(), legal)
            return sum(p * c for p, c in zip(pi, children))
        # our node: range plays the blueprint mix, split weights per action
        hist = state.history_str()
        total = 0.0
        for k, a in enumerate(legal):
            w_child = {}
            for label, w in weights.items():
                p = our_buckets.pi(label, hist, legal)[k]
                if w * p > 0:
                    w_child[label] = w * p
            if w_child:
                child_mass = sum(w_child.values())
                total += child_mass * br(state.apply(a), gidx, w_child)
        mass = sum(weights.values())
        return total / mass if mass > 0 else 0.0

    root_weights = {label: b["weight"]
                    for label, b in our_buckets.by_label.items()}
    group_cbv = {gidx: br(shadow, gidx, root_weights) for gidx in rep_rank}
    return {h: group_cbv[groups[h]] for h in holes}, groups


class SafeRiverResolver:
    """Gadget-game re-solver. Same .resolve entry shape as SubgameResolver
    but additionally needs our range and the blueprint policy/bucketer."""

    from_street = 3  # river only

    def __init__(self, iterations: int = 12000, seed: int = 0,
                 cbv_mode: str = "br"):
        # The two-range gadget needs FAR more iterations than the unsafe
        # single-hand solver: at 3k the extracted strategies are
        # mid-convergence noise (measured live: -2428 mbb/hand on river
        # hands); by 10-20k they stabilize. Do not lower this for speed.
        self.iterations = iterations
        self.cbv_mode = cbv_mode  # "br" (loose) or "blueprint" (tight)
        self.rng = random.Random(seed)

    def resolve(self, shadow, our_hole, opp_range, our_range, policy,
                bucketer) -> dict[str, float]:
        our_seat = shadow.to_act
        opp_seat = 1 - our_seat
        board = shadow.board
        root_hist = shadow.history_str()

        our_buckets = _OurBuckets(our_range, board, bucketer, policy)
        cbv, opp_group = compute_cbvs(shadow, opp_range, our_buckets, opp_seat,
                                      mode=self.cbv_mode, policy=policy,
                                      bucketer=bucketer)

        opp_holes = list(opp_range.keys())
        opp_probs = [opp_range[h] for h in opp_holes]
        our_holes = list(our_range.keys())
        our_probs = [our_range[h] for h in our_holes]
        label_of = {g: bucketer.label(g, board, 3) for g in our_holes}

        nodes: dict[str, list] = {}

        def node_for(key, actions):
            n = nodes.get(key)
            if n is None:
                n = nodes[key] = [[0.0] * len(actions),
                                  [0.0] * len(actions), list(actions)]
            return n

        def infoset_key(state, g, h):
            if state.to_act == our_seat:
                return f"u{label_of.get(g, '?')}|{state.history_str()}"
            return f"o{opp_group[h]}|{state.history_str()}"

        def traverse(state, g, h, update_player):
            if state.is_terminal():
                return state.utility(update_player)
            actions = state.legal_actions()
            regret, strat, _ = node_for(infoset_key(state, g, h), actions)
            sigma = regret_matching(regret)
            if state.to_act == update_player:
                utils = [traverse(state.apply(a), g, h, update_player)
                         for a in actions]
                value = sum(s * u for s, u in zip(sigma, utils))
                for k in range(len(actions)):
                    regret[k] += utils[k] - value
                return value
            for k in range(len(actions)):
                strat[k] += sigma[k]
            action = self.rng.choices(actions, weights=sigma)[0]
            return traverse(state.apply(action), g, h, update_player)

        def gadget(g, h, update_player):
            """Opp T/F root, then the betting subgame."""
            key = f"gadget|{opp_group[h]}"
            regret, strat, _ = node_for(key, [TERMINATE, FOLLOW])
            sigma = regret_matching(regret)
            state = shadow._clone()
            state.holes = ((g, h) if our_seat == 0 else (h, g))

            if update_player == opp_seat:
                # T pays the opponent its blueprint value; utilities here are
                # from the opponent's perspective (update player).
                u_t = cbv[h]
                u_f = traverse(state, g, h, opp_seat)
                value = sigma[0] * u_t + sigma[1] * u_f
                regret[0] += u_t - value
                regret[1] += u_f - value
                return value
            strat[0] += sigma[0]
            strat[1] += sigma[1]
            if self.rng.random() < sigma[0]:
                return -cbv[h]  # opponent terminates: our payoff is -CBV
            return traverse(state, g, h, update_player)

        for _ in range(self.iterations):
            g = self.rng.choices(our_holes, weights=our_probs)[0]
            h = self.rng.choices(opp_holes, weights=opp_probs)[0]
            if set(g) & set(h):
                continue  # card-removal rejection
            for update_player in (our_seat, opp_seat):
                gadget(g, h, update_player)

        # Play our ACTUAL hand's bucket of the range strategy.
        self.last_nodes = nodes  # exposed for the safety-property tests
        actual_label = bucketer.label(our_hole, board, 3)
        root_key = f"u{actual_label}|{root_hist}"
        actions = shadow.legal_actions()
        node = nodes.get(root_key)
        if node is None or sum(node[1]) <= 0:
            return {a: 1.0 / len(actions) for a in actions}
        total = sum(node[1])
        return {a: node[1][k] / total for k, a in enumerate(node[2])}
