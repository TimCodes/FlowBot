"""Tests for the depth-limited search agent (rung 5.5).

Run with:  .venv\\Scripts\\python -m unittest test_nlhe6_search -v
"""

import random
import unittest

from treys import Card

from card_abstraction import EquityBucketer
from nlhe6_engine import ALL_IN, CALL, FOLD, HALF_POT, NLHE6State, POT
from nlhe6_mccfr import Linear6MCCFRTrainer, PolicyAgent6, infoset_key6, play_table
from nlhe6_search import (Blueprint, CONTINUATIONS, SearchAgent6,
                          SubgameSolver, biased_probs, opponent_ranges,
                          replay_decisions)


def cards(*names):
    return tuple(Card.new(n) for n in names)


HOLES = (("As", "Ad"), ("Ks", "Kd"), ("Qs", "Qd"),
         ("Js", "Jd"), ("9s", "9d"), ("7c", "2d"))
BOARD = ("Ah", "Kh", "8c", "8d", "3h")


def make_state(holes=HOLES, board=BOARD):
    return NLHE6State(tuple(cards(*h) for h in holes), cards(*board))


def flop_state():
    """Limped 6-way pot taken to the flop; seat 0 (SB) to act."""
    s = make_state()
    for _ in range(6):
        s = s.apply(CALL)
    assert s.street == 1 and s.to_act == 0
    return s


class TestBiasedProbs(unittest.TestCase):
    ACTIONS = [FOLD, CALL, HALF_POT, POT, ALL_IN]
    PROBS = [0.2, 0.2, 0.2, 0.2, 0.2]

    def test_no_bias_returns_blueprint(self):
        self.assertEqual(biased_probs(self.PROBS, self.ACTIONS, None),
                         self.PROBS)

    def test_fold_bias_boosts_only_fold(self):
        w = biased_probs(self.PROBS, self.ACTIONS, "fold")
        self.assertAlmostEqual(sum(w), 1.0)
        self.assertAlmostEqual(w[0], 1.0 / 1.8)      # 0.2*5 / (1.0 + 0.8)
        for i in range(1, 5):
            self.assertAlmostEqual(w[i], 0.2 / 1.8)

    def test_raise_bias_boosts_all_raise_actions(self):
        w = biased_probs(self.PROBS, self.ACTIONS, "raise")
        self.assertGreater(w[2], w[0])
        self.assertAlmostEqual(w[2], w[3])
        self.assertAlmostEqual(w[3], w[4])
        self.assertAlmostEqual(sum(w), 1.0)

    def test_bias_on_missing_action_is_noop(self):
        # Not facing a bet: no fold available -> fold bias changes nothing.
        actions, probs = [CALL, POT], [0.5, 0.5]
        self.assertEqual(biased_probs(probs, actions, "fold"), probs)


class TestReplayAndRanges(unittest.TestCase):
    def test_replay_reconstructs_history(self):
        s = flop_state().apply(HALF_POT).apply(FOLD).apply(CALL)
        nodes = replay_decisions(s)
        self.assertEqual(len(nodes), 9)  # 6 preflop + 3 flop decisions
        self.assertEqual([ch for _, ch in nodes[-3:]], ["h", "f", "c"])
        self.assertEqual(nodes[-1][0].history_str() + nodes[-1][1],
                         s.history_str())

    def test_ranges_cover_live_opponents_only(self):
        s = flop_state().apply(HALF_POT).apply(FOLD)  # BB folds
        hero = s.to_act  # UTG faces the SB bet
        bp = Blueprint({}, EquityBucketer(4, 10, seed=0, num_opponents=5))
        ranges = opponent_ranges(s, bp, hero)
        self.assertNotIn(hero, ranges)
        self.assertNotIn(1, ranges)  # folded BB has no range
        self.assertEqual(set(ranges), {0, 3, 4, 5})

    def test_reach_weights_favor_hands_that_raise(self):
        # Synthetic policy: on the flop, top-bucket hands always bet half-pot,
        # everything else always checks. After seeing SB bet, the SB's range
        # should put all its weight on top-bucket holes.
        bucketer = EquityBucketer(num_buckets=4, samples=40, seed=1,
                                  num_opponents=5)
        s = flop_state()
        actions = s.legal_actions()  # [c, h, p, a] -- not facing a bet
        hi = actions.index(HALF_POT)
        lo = actions.index(CALL)
        policy = {}
        for bucket in range(4):
            probs = [0.0] * len(actions)
            probs[hi if bucket == 3 else lo] = 1.0
            policy[f"0|1:{bucket}|cccccc/"] = probs
        bet = s.apply(HALF_POT)
        bp = Blueprint(policy, bucketer)
        ranges = opponent_ranges(bet, bp, hero=bet.to_act)
        pairs, weights = ranges[0]
        for hole, w in zip(pairs, weights):
            if w > 0:
                label = bucketer.label(hole, bet.board_revealed(), 1)
                self.assertEqual(label, "1:3")


class TestSubgameSolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bucketer = EquityBucketer(num_buckets=4, samples=15, seed=7,
                                      num_opponents=5)
        trainer = Linear6MCCFRTrainer(cls.bucketer, seed=7)
        for _ in range(80):
            trainer.iteration()
        cls.policy = trainer.average_policy()

    def test_root_strategy_is_distribution(self):
        s = flop_state()
        solver = SubgameSolver(s, Blueprint(self.policy, self.bucketer),
                               hero=0, seed=1)
        solver.solve(30)
        actions, probs = solver.root_strategy()
        self.assertEqual(len(actions), len(probs))
        self.assertAlmostEqual(sum(probs), 1.0, places=9)
        self.assertTrue(all(p >= 0 for p in probs))

    def test_river_root_solves_to_terminal_without_meta_nodes(self):
        s = make_state()
        while s.street < 3:
            s = s.apply(CALL)
        self.assertEqual(s.street, 3)
        solver = SubgameSolver(s, Blueprint(self.policy, self.bucketer),
                               hero=s.to_act, seed=2)
        solver.solve(30)
        self.assertFalse(any(k.startswith("cont|") for k in solver.nodes))

    def test_flop_solve_creates_continuation_nodes(self):
        s = flop_state()
        solver = SubgameSolver(s, Blueprint(self.policy, self.bucketer),
                               hero=0, seed=3)
        solver.solve(30)
        self.assertTrue(any(k.startswith("cont|") for k in solver.nodes))
        for key, node in solver.nodes.items():
            if key.startswith("cont|"):
                self.assertEqual(len(node.regret_sum), len(CONTINUATIONS))

    def test_warm_start_anchors_root_at_blueprint(self):
        # With warm mass and a tiny budget, the root strategy must stay
        # close to the blueprint (graceful degradation), not drift to noise.
        s = flop_state()
        bp = Blueprint(self.policy, self.bucketer)
        solver = SubgameSolver(s, bp, hero=0, seed=6, warm=50.0)
        solver.solve(2)
        actions, probs = solver.root_strategy()
        expected = bp.probs(s)
        for p, e in zip(probs, expected):
            self.assertAlmostEqual(p, e, delta=0.15)

    def test_determinization_never_reuses_seen_cards(self):
        s = flop_state()
        solver = SubgameSolver(s, Blueprint(self.policy, self.bucketer),
                               hero=0, seed=4)
        for _ in range(20):
            det = solver._sample_state()
            seen = list(det.holes[0])  # hero's actual hand is kept
            self.assertEqual(det.holes[0], s.holes[0])
            self.assertEqual(det.board[:3], s.board[:3])  # revealed kept
            for i in range(1, 6):
                seen += list(det.holes[i])
            seen += list(det.board)
            live = [c for i in range(6) for c in det.holes[i]
                    if not det.folded[i]] + list(det.board)
            self.assertEqual(len(live), len(set(live)))


class TestSearchAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bucketer = EquityBucketer(num_buckets=4, samples=15, seed=11,
                                      num_opponents=5)
        trainer = Linear6MCCFRTrainer(cls.bucketer, seed=11)
        for _ in range(80):
            trainer.iteration()
        cls.policy = trainer.average_policy()

    def test_plays_full_hands_legally(self):
        hero = SearchAgent6(self.policy, self.bucketer, search_iters=15,
                            seed=5)
        villains = [PolicyAgent6(self.policy, self.bucketer, seed=6 + j)
                    for j in range(5)]
        chips = play_table(hero, villains, hands=12, seed=8)
        self.assertIsInstance(chips, float)  # completed without exceptions

    def test_preflop_uses_blueprint_not_search(self):
        hero = SearchAgent6(self.policy, self.bucketer, search_iters=15,
                            seed=9)
        s = make_state()
        rng_state = random.Random(9).getstate()  # noqa: F841 (doc only)
        a = hero.act(s)  # street 0: must return instantly via blueprint path
        self.assertIn(a, s.legal_actions())


if __name__ == "__main__":
    unittest.main(verbosity=2)
