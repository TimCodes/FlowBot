"""Tests for the no-limit engine and the HUNL blueprint trainer.

Run with:  .venv\\Scripts\\python -m unittest test_nlhe -v
"""

import unittest

from treys import Card

from card_abstraction import EquityBucketer
from hulhe_mccfr import ESMCCFRTrainer, PolicyAgent, RandomAgent, play_match
from hunl_blueprint import deal_nlhe
from nlhe_engine import (ALL_IN, BIG_BLIND, CALL, DOUBLE_POT, FOLD, HALF_POT,
                         NLHEState, NLHEStateX, POT, SMALL_BLIND, STACK)


def cards(*names):
    return tuple(Card.new(n) for n in names)


def make_state(h0=("As", "Ad"), h1=("7c", "2d"),
               board=("Ks", "Qh", "Jd", "8c", "3h")):
    return NLHEState((cards(*h0), cards(*h1)), cards(*board))


class TestNLHEEngine(unittest.TestCase):
    def test_initial_state(self):
        s = make_state()
        self.assertEqual(s.contrib, [SMALL_BLIND, BIG_BLIND])
        self.assertEqual(s.to_act, 0)
        self.assertEqual(s.legal_actions(), [FOLD, CALL, HALF_POT, POT, ALL_IN])

    def test_preflop_raise_sizes(self):
        s = make_state()
        # Pot after call = 200: half-pot raise -> to 200 (a min-raise),
        # pot raise -> to 300 (the classic 3x open).
        self.assertEqual(s.raise_to_amount(HALF_POT), 200)
        self.assertEqual(s.raise_to_amount(POT), 300)
        self.assertEqual(s.raise_to_amount(ALL_IN), STACK)

    def test_pot_raise_after_raise(self):
        s = make_state().apply(POT)          # SB raises to 300
        self.assertEqual(s.contrib, [300, 100])
        self.assertEqual(s.last_raise, 200)
        # BB pot-raise: call 200 more -> pot 600, raise by 600 -> to 900.
        self.assertEqual(s.raise_to_amount(POT), 900)

    def test_min_raise_prunes_undersized_half_pot(self):
        s = make_state().apply(POT)          # last_raise = 200
        # BB half-pot raise would be to 600 (>= min-raise 500): legal.
        self.assertIn(HALF_POT, s.legal_actions())
        self.assertEqual(s.raise_to_amount(HALF_POT), 600)

    def test_all_in_and_call_runs_out_board(self):
        s = make_state().apply(ALL_IN).apply(CALL)
        self.assertTrue(s.is_terminal())
        self.assertEqual(s.contrib, [STACK, STACK])
        self.assertEqual(s.utility(0), STACK)   # AA holds vs 72o on this board
        self.assertEqual(len(s.board_revealed()), 5)

    def test_facing_all_in_no_reraise(self):
        s = make_state().apply(ALL_IN)
        self.assertEqual(s.legal_actions(), [FOLD, CALL])

    def test_fold_to_all_in(self):
        s = make_state().apply(ALL_IN).apply(FOLD)
        self.assertEqual(s.utility(0), 100)  # BB folded their blind

    def test_postflop_half_pot_bet(self):
        s = make_state().apply(CALL).apply(CALL)   # limp, check -> flop
        self.assertEqual(s.street, 1)
        self.assertEqual(s.to_act, 1)
        s2 = s.apply(HALF_POT)                     # BB bets 100 into 200
        self.assertEqual(s2.contrib, [100, 200])

    def test_raise_collapse_near_stack_becomes_all_in(self):
        s = make_state()
        for _ in range(4):                         # pot-raise war escalates
            if POT in s.legal_actions():
                s = s.apply(POT)
            else:
                break
        # Eventually only fold/call/all-in remain -- no dangling h/p actions
        # whose amounts round into the stack.
        for a in s.legal_actions():
            if a not in (FOLD, CALL, ALL_IN):
                self.assertLess(s.raise_to_amount(a), STACK)

    def test_checked_down_showdown_payoff(self):
        s = make_state()
        for _ in range(8):
            s = s.apply(CALL)
        self.assertTrue(s.is_terminal())
        self.assertEqual(s.utility(0), BIG_BLIND)

    def test_history_string(self):
        s = make_state().apply(POT).apply(CALL).apply(HALF_POT).apply(FOLD)
        self.assertEqual(s.history_str(), "pc/hf")


class TestExtendedProfile(unittest.TestCase):
    def make_x(self):
        return NLHEStateX((cards("As", "Ad"), cards("7c", "2d")),
                          cards("Ks", "Qh", "Jd", "8c", "3h"))

    def test_preflop_actions_include_overbet(self):
        s = self.make_x()
        self.assertEqual(s.legal_actions(),
                         [FOLD, CALL, HALF_POT, POT, DOUBLE_POT, ALL_IN])
        # Pot after call = 200: 2x-pot raise -> to 100 + 400 = 500 (5x open).
        self.assertEqual(s.raise_to_amount(DOUBLE_POT), 500)

    def test_std_profile_unchanged_by_subclass(self):
        s = NLHEState((cards("As", "Ad"), cards("7c", "2d")),
                      cards("Ks", "Qh", "Jd", "8c", "3h"))
        self.assertNotIn(DOUBLE_POT, s.legal_actions())

    def test_clone_preserves_profile(self):
        s = self.make_x().apply(DOUBLE_POT)
        self.assertIsInstance(s, NLHEStateX)
        self.assertEqual(s.contrib, [500, 100])
        self.assertEqual(s.history_str(), "d")
        self.assertIn(DOUBLE_POT, s.apply(CALL).legal_actions())

    def test_overbet_collapses_to_all_in_near_stack(self):
        s = self.make_x()
        for _ in range(3):  # overbet war: 500 -> 2900 -> 14900 -> cap
            if DOUBLE_POT in s.legal_actions():
                s = s.apply(DOUBLE_POT)
        for a in s.legal_actions():
            if a not in (FOLD, CALL, ALL_IN):
                self.assertLess(s.raise_to_amount(a), STACK)

    def test_payoffs_identical_across_profiles(self):
        for cls in (NLHEState, NLHEStateX):
            s = cls((cards("As", "Ad"), cards("7c", "2d")),
                    cards("Ks", "Qh", "Jd", "8c", "3h"))
            for _ in range(8):
                s = s.apply(CALL)
            self.assertEqual(s.utility(0), BIG_BLIND)


class TestBlueprintTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bucketer = EquityBucketer(num_buckets=8, samples=30, seed=9)
        cls.trainer = ESMCCFRTrainer(cls.bucketer, seed=9,
                                     state_factory=deal_nlhe)
        for _ in range(600):
            cls.trainer.iteration()
        cls.policy = cls.trainer.average_policy()

    def test_builds_infosets(self):
        self.assertGreater(len(self.trainer.nodes), 300)

    def test_policy_is_normalized(self):
        for probs in self.policy.values():
            self.assertAlmostEqual(sum(probs), 1.0, places=9)

    def test_beats_random_agent(self):
        agent = PolicyAgent(self.policy, self.bucketer, seed=1)
        chips = play_match(agent, RandomAgent(seed=2), hands=2000, seed=3,
                           state_factory=deal_nlhe)
        self.assertGreater(chips, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
