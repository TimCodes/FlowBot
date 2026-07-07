"""Tests for the 6-max engine, multiway abstraction, and blueprint trainer.

Run with:  .venv\\Scripts\\python -m unittest test_nlhe6 -v
"""

import unittest

from treys import Card

from card_abstraction import EquityBucketer
from nlhe6_engine import (ALL_IN, BIG_BLIND, CALL, FOLD, HALF_POT, NLHE6State,
                          POT, SMALL_BLIND, STACK, settle)
from nlhe6_mccfr import (CallAgent, Linear6MCCFRTrainer, PolicyAgent6,
                         RandomAgent, deal_nlhe6, play_table)


def cards(*names):
    return tuple(Card.new(n) for n in names)


HOLES = (("As", "Ad"), ("Ks", "Kd"), ("Qs", "Qd"),
         ("Js", "Jd"), ("9s", "9d"), ("7c", "2d"))
BOARD = ("Ah", "Kh", "8c", "8d", "3h")  # AA makes top set on this runout


def make_state(holes=HOLES, board=BOARD):
    return NLHE6State(tuple(cards(*h) for h in holes), cards(*board))


def fold_out(state, keep):
    """Fold every seat outside `keep` as action reaches them."""
    while not state.is_terminal():
        if state.to_act in keep:
            return state
        state = state.apply(FOLD)
    return state


class TestSettle(unittest.TestCase):
    def test_simple_headsup_pot(self):
        # Two players, equal contributions, seat 0 wins.
        self.assertEqual(settle([100, 100], set(), {0: 1, 1: 2}), [100, -100])

    def test_three_way_side_pots(self):
        # Classic layered all-in: short stack 100, mid 300, big 300.
        # Seat 0 (short) has the best hand: wins only the 300 main pot;
        # seat 1 beats seat 2 for the 400 side pot.
        payoff = settle([100, 300, 300], set(), {0: 1, 1: 2, 2: 3})
        self.assertEqual(payoff, [200, 100, -300])
        self.assertEqual(sum(payoff), 0)

    def test_dead_money_goes_to_winner(self):
        # Seat 2 folded after posting 100; seat 1 wins everything.
        payoff = settle([500, 500, 100], {2}, {0: 5, 1: 2})
        self.assertEqual(payoff, [-500, 600, -100])

    def test_folded_bigger_stack_layer_returns_to_live_top(self):
        # Seat 0 bet 300 then folded to a raise; sole live seat 1 takes all.
        payoff = settle([300, 900, 0], {0, 2}, {1: 7})
        self.assertEqual(payoff, [-300, 300, 0])

    def test_split_pot_odd_chip_to_earliest_seat(self):
        payoff = settle([3, 3, 1], {2}, {0: 4, 1: 4})
        self.assertEqual(sum(payoff), 0)
        self.assertEqual(payoff, [1, 0, -1])  # seat 0 gets the odd chip


class TestNLHE6Engine(unittest.TestCase):
    def test_initial_state(self):
        s = make_state()
        self.assertEqual(s.contrib, [SMALL_BLIND, BIG_BLIND, 0, 0, 0, 0])
        self.assertEqual(s.to_act, 2)  # UTG opens
        self.assertEqual(s.legal_actions(),
                         [FOLD, CALL, HALF_POT, POT, ALL_IN])

    def test_preflop_open_sizes(self):
        s = make_state()
        # UTG facing 100: pot after call = 50+100+100+100 = 350... no:
        # contribs sum 150, to_call 100 -> pot_after_call 250.
        # half-pot raise-by 125 -> to 225; pot raise-by 250 -> to 350.
        self.assertEqual(s.raise_to_amount(HALF_POT), 225)
        self.assertEqual(s.raise_to_amount(POT), 350)
        self.assertEqual(s.raise_to_amount(ALL_IN), STACK)

    def test_fold_around_to_bb_wins_blinds(self):
        s = make_state()
        for _ in range(5):  # UTG..SB all fold
            s = s.apply(FOLD)
        self.assertTrue(s.is_terminal())
        p = s.payoffs()
        self.assertEqual(p[1], SMALL_BLIND)   # BB collects the SB
        self.assertEqual(p[0], -SMALL_BLIND)
        self.assertEqual(p[2:], [0, 0, 0, 0])
        self.assertEqual(sum(p), 0)

    def test_bb_option_after_limps(self):
        s = make_state()
        for _ in range(5):  # UTG..BTN limp, SB completes
            s = s.apply(CALL)
        self.assertFalse(s.is_terminal())
        self.assertEqual(s.to_act, 1)  # BB still owes an action
        self.assertIn(POT, s.legal_actions())  # and may raise the option
        flop = s.apply(CALL)  # BB checks
        self.assertEqual(flop.street, 1)
        self.assertEqual(flop.to_act, 0)  # SB first postflop
        self.assertEqual(sum(flop.contrib), 6 * BIG_BLIND)

    def test_raise_reopens_action(self):
        s = make_state()
        for _ in range(4):  # UTG, HJ, CO, BTN limp
            s = s.apply(CALL)
        s = s.apply(POT)  # SB squeezes
        # Everyone who already limped owes another action before the flop.
        self.assertEqual(s.to_act, 1)
        for _ in range(5):  # BB + four limpers call the squeeze
            self.assertFalse(s.is_terminal())
            s = s.apply(CALL)
        self.assertEqual(s.street, 1)

    def test_min_raise_enforced(self):
        s = make_state().apply(POT)  # UTG raises to 350 (raise-by 250)
        self.assertEqual(s.contrib[2], 350)
        self.assertEqual(s.last_raise, 250)
        nxt = s.legal_actions()
        for a in nxt:
            if a in (HALF_POT, POT):
                self.assertGreaterEqual(s.raise_to_amount(a), 350 + 250)

    def test_all_in_call_fastforwards_board(self):
        s = make_state().apply(ALL_IN)      # UTG jams
        s = fold_out(s, keep={1})           # HJ..SB fold to the BB
        self.assertEqual(s.legal_actions(), [FOLD, CALL])  # no re-raise
        s = s.apply(CALL)
        self.assertTrue(s.is_terminal())
        p = s.payoffs()
        # AA (seat 0 hole) belongs to seat 0... here the jammer is seat 2
        # with QQ vs BB's KK on an AK8/8/3 board: KK wins main + dead SB.
        self.assertEqual(p[1], STACK + SMALL_BLIND)
        self.assertEqual(p[2], -STACK)
        self.assertEqual(sum(p), 0)

    def test_checked_down_showdown_multiway(self):
        s = make_state()
        while not s.is_terminal():
            s = s.apply(CALL)
        p = s.payoffs()
        # Seat 0 (AA, top set) wins 5 * BB from the other five seats.
        self.assertEqual(p[0], 5 * BIG_BLIND)
        self.assertEqual(sum(p), 0)

    def test_history_string_multiway(self):
        s = make_state().apply(POT).apply(FOLD).apply(FOLD).apply(FOLD)
        s = s.apply(FOLD).apply(CALL)  # SB folds, BB calls
        self.assertEqual(s.street, 1)
        self.assertTrue(s.history_str().startswith("pffffc/"))

    def test_utility_matches_payoffs(self):
        s = make_state()
        for _ in range(5):
            s = s.apply(FOLD)
        for seat in range(6):
            self.assertEqual(s.utility(seat), s.payoffs()[seat])

    def test_postflop_first_live_seat_acts(self):
        s = make_state()
        s = s.apply(CALL)   # UTG limps
        s = s.apply(FOLD).apply(FOLD).apply(FOLD)  # HJ, CO, BTN fold
        s = s.apply(FOLD)   # SB folds
        s = s.apply(CALL)   # BB checks option -> flop
        self.assertEqual(s.street, 1)
        self.assertEqual(s.to_act, 1)  # SB folded, BB is first live seat


class TestMultiwayAbstraction(unittest.TestCase):
    def test_multiway_equity_ranks_hands(self):
        b = EquityBucketer(num_buckets=8, samples=400, seed=3,
                           num_opponents=5)
        aa = b.hand_strength(cards("As", "Ad"), ())
        junk = b.hand_strength(cards("7c", "2d"), ())
        self.assertGreater(aa, 0.35)   # ~0.49 five-way
        self.assertLess(junk, 0.15)    # ~0.07 five-way
        self.assertGreater(aa, junk)

    def test_multiway_equity_below_headsup_equity(self):
        hu = EquityBucketer(num_buckets=8, samples=400, seed=3)
        mw = EquityBucketer(num_buckets=8, samples=400, seed=3,
                            num_opponents=5)
        hole = cards("As", "Ad")
        self.assertLess(mw.hand_strength(hole, ()),
                        hu.hand_strength(hole, ()))


class TestBlueprintTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bucketer = EquityBucketer(num_buckets=8, samples=25, seed=9,
                                      num_opponents=5)
        cls.trainer = Linear6MCCFRTrainer(cls.bucketer, seed=9)
        for _ in range(300):
            cls.trainer.iteration()
        cls.policy = cls.trainer.average_policy()

    def test_builds_infosets(self):
        self.assertGreater(len(self.trainer.nodes), 1000)

    def test_policy_is_normalized(self):
        for probs in self.policy.values():
            self.assertAlmostEqual(sum(probs), 1.0, places=9)

    def test_linear_weighting_advances(self):
        self.assertEqual(self.trainer.t, 300)

    def test_beats_random_table(self):
        agent = PolicyAgent6(self.policy, self.bucketer, seed=1)
        chips = play_table(agent, [RandomAgent(seed=2 + j) for j in range(5)],
                           hands=1200, seed=3)
        self.assertGreater(chips, 0.0)

    def test_pruning_skips_bad_actions(self):
        b = EquityBucketer(num_buckets=4, samples=10, seed=5,
                           num_opponents=5)
        t = Linear6MCCFRTrainer(b, seed=5, prune_after=5,
                                prune_threshold=-1.0, regret_floor=-10.0)
        for _ in range(60):
            t.iteration()
        floors = [r for n in t.nodes.values() for r in n.regret_sum]
        self.assertGreaterEqual(min(floors), -10.0)  # floor respected


class TestTableRunner(unittest.TestCase):
    def test_call_stations_conserve_chips(self):
        hero = CallAgent()
        total = play_table(hero, [CallAgent() for _ in range(5)],
                           hands=600, seed=11)
        # Checked-down tables are ~zero-sum for a rotating seat; equity noise
        # over 600 hands stays well inside a big blind per hand.
        self.assertLess(abs(total), BIG_BLIND)

    def test_deal_produces_distinct_cards(self):
        import random
        from holdem_engine import FULL_DECK
        rng = random.Random(0)
        s = deal_nlhe6(rng.sample(FULL_DECK, 17))
        seen = [c for h in s.holes for c in h] + list(s.board)
        self.assertEqual(len(seen), len(set(seen)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
