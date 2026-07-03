"""Tests for the HULHE engine, card abstraction, and MCCFR trainer.

Run with:  .venv\\Scripts\\python -m unittest test_hulhe -v
"""

import unittest

from treys import Card

from card_abstraction import EquityBucketer, preflop_class
from holdem_engine import CALL, FOLD, HULHEState, RAISE
from hulhe_mccfr import (CallAgent, ESMCCFRTrainer, PolicyAgent, RandomAgent,
                         play_match)


def cards(*names):
    return tuple(Card.new(n) for n in names)


def make_state(h0=("As", "Ad"), h1=("7c", "2d"),
               board=("Ks", "Qh", "Jd", "8c", "3h")):
    return HULHEState((cards(*h0), cards(*h1)), cards(*board))


class TestEngine(unittest.TestCase):
    def test_initial_state(self):
        s = make_state()
        self.assertEqual(s.contrib, [1, 2])
        self.assertEqual(s.to_act, 0)
        self.assertEqual(s.legal_actions(), [FOLD, CALL, RAISE])

    def test_sb_fold_loses_small_blind(self):
        s = make_state().apply(FOLD)
        self.assertTrue(s.is_terminal())
        self.assertEqual(s.utility(0), -1)
        self.assertEqual(s.utility(1), 1)

    def test_limp_gives_bb_option(self):
        s = make_state().apply(CALL)  # SB limps
        self.assertFalse(s.is_terminal())
        self.assertEqual(s.street, 0)  # BB still has the option
        self.assertEqual(s.to_act, 1)
        self.assertEqual(s.contrib, [2, 2])
        self.assertNotIn(FOLD, s.legal_actions())  # not facing a bet

    def test_limp_check_advances_to_flop(self):
        s = make_state().apply(CALL).apply(CALL)
        self.assertEqual(s.street, 1)
        self.assertEqual(s.to_act, 1)  # BB first postflop
        self.assertEqual(len(s.board_revealed()), 3)
        self.assertEqual(s.bets, 0)

    def test_raise_call_preflop(self):
        s = make_state().apply(RAISE).apply(CALL)
        self.assertEqual(s.street, 1)
        self.assertEqual(s.contrib, [4, 4])

    def test_bet_cap_enforced(self):
        s = make_state()
        for _ in range(3):  # BB blind bet + 3 raises = cap of 4
            self.assertIn(RAISE, s.legal_actions())
            s = s.apply(RAISE)
        self.assertNotIn(RAISE, s.legal_actions())
        self.assertEqual(s.legal_actions(), [FOLD, CALL])

    def test_postflop_bet_sizes(self):
        s = make_state().apply(CALL).apply(CALL)      # to flop, pot 2+2
        s = s.apply(RAISE)                            # BB bets 2 (small bet)
        self.assertEqual(s.contrib, [2, 4])
        s = s.apply(CALL).apply(CALL).apply(CALL)     # call; turn check-check
        self.assertEqual(s.street, 3)
        s = s.apply(RAISE)                            # river big bet = 4
        self.assertEqual(s.contrib, [4, 8])

    def test_showdown_best_hand_wins(self):
        s = make_state()  # AA vs 72o on K-Q-J-8-3: aces win
        for _ in range(8):  # check/call to showdown
            s = s.apply(CALL)
        self.assertTrue(s.is_terminal())
        self.assertEqual(s.utility(0), 2)   # wins BB's 2 chips
        self.assertEqual(s.utility(1), -2)

    def test_showdown_tie_splits(self):
        # Board plays: royal flush on board.
        s = make_state(h0=("2c", "3d"), h1=("4h", "5s"),
                       board=("As", "Ks", "Qs", "Js", "Ts"))
        for _ in range(8):
            s = s.apply(CALL)
        self.assertEqual(s.utility(0), 0)

    def test_fold_to_river_bet(self):
        s = make_state()
        for _ in range(6):  # to the river, checked down
            s = s.apply(CALL)
        self.assertEqual(s.street, 3)
        s = s.apply(RAISE)  # BB bets 4 on the river
        s = s.apply(FOLD)   # SB folds having contributed 2
        self.assertEqual(s.utility(0), -2)

    def test_history_string(self):
        s = make_state().apply(RAISE).apply(CALL).apply(CALL).apply(RAISE)
        self.assertEqual(s.history_str(), "rc/cr")


class TestAbstraction(unittest.TestCase):
    def test_preflop_classes(self):
        self.assertEqual(preflop_class(cards("Ah", "Kh")), "AKs")
        self.assertEqual(preflop_class(cards("Kd", "Ah")), "AKo")
        self.assertEqual(preflop_class(cards("7c", "7d")), "77")
        self.assertEqual(preflop_class(cards("2c", "9h")), "92o")

    def test_hand_strength_ordering(self):
        b = EquityBucketer(num_buckets=8, samples=400, seed=3)
        aa = b.hand_strength(cards("As", "Ad"), ())
        seven_deuce = b.hand_strength(cards("7c", "2d"), ())
        self.assertGreater(aa, 0.80)
        self.assertLess(seven_deuce, 0.45)
        self.assertGreater(aa, seven_deuce)

    def test_nut_hand_maxes_bucket(self):
        b = EquityBucketer(num_buckets=8, samples=200, seed=3)
        label = b.label(cards("As", "Ks"), cards("Qs", "Js", "Ts"), 1)
        self.assertEqual(label, "1:7")  # royal flush -> top bucket

    def test_cache_reuse(self):
        b = EquityBucketer(samples=50, seed=3)
        h, board = cards("As", "Ad"), cards("Ks", "Qh", "Jd")
        first = b.hand_strength(h, board)
        self.assertEqual(b.hand_strength(h, board), first)
        self.assertEqual(len(b.cache), 1)


class TestTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bucketer = EquityBucketer(num_buckets=8, samples=30, seed=5)
        cls.trainer = ESMCCFRTrainer(cls.bucketer, seed=5)
        for _ in range(600):
            cls.trainer.iteration()
        cls.policy = cls.trainer.average_policy()

    def test_builds_infosets(self):
        self.assertGreater(len(self.trainer.nodes), 200)

    def test_policy_is_normalized(self):
        for probs in self.policy.values():
            self.assertAlmostEqual(sum(probs), 1.0, places=9)
            self.assertTrue(all(p >= 0 for p in probs))

    def test_beats_random_agent(self):
        agent = PolicyAgent(self.policy, self.bucketer, seed=1)
        chips = play_match(agent, RandomAgent(seed=2), hands=2000, seed=3)
        self.assertGreater(chips, 0.0)

    def test_match_runner_is_fair_for_identical_agents(self):
        a, b = RandomAgent(seed=1), RandomAgent(seed=2)
        chips = play_match(a, b, hands=4000, seed=4)
        self.assertLess(abs(chips), 0.35)  # symmetric matchup ~ 0


if __name__ == "__main__":
    unittest.main(verbosity=2)
