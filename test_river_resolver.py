"""Tests for river subgame re-solving.

Run with:  .venv\\Scripts\\python -m unittest test_river_resolver -v
"""

import unittest

from treys import Card

from card_abstraction import EquityBucketer
from nlhe_engine import ALL_IN, CALL, FOLD, HALF_POT, NLHEState, POT
from river_resolver import RiverResolver, SubgameResolver, opponent_range
from slumbot_client import replay_abstract


def cards(*names):
    return tuple(Card.new(n) for n in names)


def river_state(hole, board):
    """Checked-down state at the start of river betting (to_act = seat 1)."""
    s = NLHEState((cards(*hole), cards(*hole)), cards(*board))
    for _ in range(6):  # limp-check, check-check, check-check
        s = s.apply(CALL)
    assert s.street == 3 and s.to_act == 1
    return s


def turn_state(hole, board):
    """Checked-down state at the start of turn betting (to_act = seat 1).

    board[4] is a placeholder the resolver replaces with sampled rivers.
    """
    s = NLHEState((cards(*hole), cards(*hole)), cards(*board))
    for _ in range(4):  # limp-check, flop check-check
        s = s.apply(CALL)
    assert s.street == 2 and s.to_act == 1
    return s


def uniform_range(hole, board):
    bucketer = EquityBucketer(samples=10, seed=2)
    return opponent_range({}, bucketer, [], 0, cards(*hole), cards(*board))


NUTS_BOARD = ("Qs", "Js", "Ts", "2c", "2d")


class TestOpponentRange(unittest.TestCase):
    def test_uniform_when_no_decisions(self):
        rng = uniform_range(("As", "Ks"), NUTS_BOARD)
        self.assertEqual(len(rng), 45 * 44 // 2)  # C(45,2) unblocked combos
        self.assertAlmostEqual(sum(rng.values()), 1.0, places=9)

    def test_blocked_cards_excluded(self):
        rng = uniform_range(("As", "Ks"), NUTS_BOARD)
        blocked = set(cards("As", "Ks", *NUTS_BOARD))
        for hole in rng:
            self.assertFalse(set(hole) & blocked)

    def test_trace_recording(self):
        trace = []
        replay_abstract("b300c/", ["As", "Ks"], ["Qs", "Js", "Ts"],
                        trace_out=trace)
        self.assertEqual(len(trace), 2)
        seats = [t[0] for t in trace]
        chosen = [t[4] for t in trace]
        self.assertEqual(seats, [0, 1])     # SB raised, BB called
        self.assertEqual(chosen, ["p", "c"])
        self.assertEqual(trace[0][1], 0)    # preflop street

    def test_range_respects_blueprint_zeroes(self):
        # Synthetic blueprint: at the SB's opening infoset, every preflop
        # class folds except pairs of aces, which always pot-raise. After
        # observing a pot raise, the range must be exactly {AA combos}.
        bucketer = EquityBucketer(samples=10, seed=2)
        legal = ("f", "c", "h", "p", "a")
        policy = {}
        from itertools import combinations
        from card_abstraction import preflop_class
        from holdem_engine import FULL_DECK
        for c1, c2 in combinations(FULL_DECK, 2):
            cls = preflop_class((c1, c2))
            key = f"{cls}|"
            policy[key] = ([0, 0, 0, 1, 0] if cls == "AA"
                           else [1, 0, 0, 0, 0])
        trace = [(0, 0, "", legal, "p")]
        rng = opponent_range(policy, bucketer, trace, 0,
                             cards("Ks", "Kd"), cards(*NUTS_BOARD))
        self.assertTrue(rng)
        for hole in rng:
            self.assertEqual(preflop_class(hole), "AA")


class TestRiverResolver(unittest.TestCase):
    def test_returns_normalized_distribution(self):
        s = river_state(("9h", "8h"), NUTS_BOARD)
        rng = uniform_range(("9h", "8h"), NUTS_BOARD)
        dist = RiverResolver(iterations=500, seed=1).resolve(
            s, cards("9h", "8h"), rng)
        self.assertEqual(set(dist), set(s.legal_actions()))
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)
        self.assertNotIn(FOLD, dist)  # not facing a bet

    def test_nuts_facing_bet_never_folds_usually_raises(self):
        s = river_state(("As", "Ks"), NUTS_BOARD).apply(HALF_POT)
        self.assertEqual(s.to_act, 0)  # we act facing the bet
        rng = uniform_range(("As", "Ks"), NUTS_BOARD)
        dist = RiverResolver(iterations=3000, seed=1).resolve(
            s, cards("As", "Ks"), rng)
        self.assertLess(dist.get(FOLD, 0.0), 0.05)
        raise_mass = sum(dist.get(a, 0.0) for a in (HALF_POT, POT, ALL_IN))
        self.assertGreater(raise_mass, 0.4)

    def test_air_facing_bet_mostly_folds(self):
        # 4h3h on Qs Js Ts 2c 2d loses to essentially the entire range.
        s = river_state(("4h", "3h"), NUTS_BOARD).apply(HALF_POT)
        rng = uniform_range(("4h", "3h"), NUTS_BOARD)
        dist = RiverResolver(iterations=3000, seed=1).resolve(
            s, cards("4h", "3h"), rng)
        self.assertGreater(dist.get(FOLD, 0.0), 0.4)
        self.assertLess(dist.get(CALL, 0.0), 0.4)


class TestTurnResolver(unittest.TestCase):
    # Range built on the 4 revealed cards only; board[4] is a dummy the
    # resolver never treats as known.
    TURN_BOARD4 = ("Qs", "Js", "Ts", "2c")

    def test_returns_normalized_distribution_from_turn(self):
        s = turn_state(("9h", "8h"), self.TURN_BOARD4 + ("2d",))
        rng = uniform_range(("9h", "8h"), self.TURN_BOARD4)
        dist = SubgameResolver(iterations=800, seed=1, from_street=2).resolve(
            s, cards("9h", "8h"), rng)
        self.assertEqual(set(dist), set(s.legal_actions()))
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)
        self.assertNotIn(FOLD, dist)

    def test_made_royal_never_folds_turn_bet(self):
        # Note: vs a single half-pot bet the solver overwhelmingly *calls*
        # (trapping -- keeping bluffs in for the river), which is sound; the
        # only watertight assertion here is that it never folds.
        s = turn_state(("As", "Ks"),
                       self.TURN_BOARD4 + ("2d",)).apply(HALF_POT)
        self.assertEqual(s.to_act, 0)  # we face the turn bet
        rng = uniform_range(("As", "Ks"), self.TURN_BOARD4)
        dist = SubgameResolver(iterations=3000, seed=1, from_street=2).resolve(
            s, cards("As", "Ks"), rng)
        self.assertLess(dist.get(FOLD, 0.0), 0.02)

    def test_made_royal_calls_turn_all_in(self):
        # Facing an all-in with an unbeatable hand, calling strictly
        # dominates: the distribution must be essentially pure call.
        s = turn_state(("As", "Ks"), self.TURN_BOARD4 + ("2d",)).apply(ALL_IN)
        self.assertEqual(s.legal_actions(), [FOLD, CALL])
        rng = uniform_range(("As", "Ks"), self.TURN_BOARD4)
        dist = SubgameResolver(iterations=2000, seed=1, from_street=2).resolve(
            s, cards("As", "Ks"), rng)
        self.assertGreater(dist[CALL], 0.95)

    def test_air_facing_turn_bet_mostly_folds(self):
        s = turn_state(("4h", "3h"),
                       self.TURN_BOARD4 + ("2d",)).apply(HALF_POT)
        rng = uniform_range(("4h", "3h"), self.TURN_BOARD4)
        dist = SubgameResolver(iterations=3000, seed=1, from_street=2).resolve(
            s, cards("4h", "3h"), rng)
        self.assertGreater(dist.get(FOLD, 0.0), 0.35)

    def test_river_only_alias_unchanged(self):
        self.assertIs(RiverResolver, SubgameResolver)
        self.assertEqual(SubgameResolver().from_street, 3)


class TestResolverPotCap(unittest.TestCase):
    NUTS_BOARD = ("Qs", "Js", "Ts", "2c", "2d")

    def _ext_river(self, hole):
        from nlhe_engine import NLHEStateX
        s = NLHEStateX((cards(*hole), cards(*hole)), cards(*self.NUTS_BOARD))
        for _ in range(6):
            s = s.apply(CALL)
        assert s.street == 3 and s.to_act == 1
        return s

    def test_ext_river_offers_overbet_without_cap(self):
        from nlhe_engine import DOUBLE_POT
        s = self._ext_river(("As", "Ks"))
        self.assertIn(DOUBLE_POT, s.legal_actions())

    def test_capped_resolver_never_returns_overbet(self):
        from nlhe_engine import DOUBLE_POT
        s = self._ext_river(("As", "Ks"))
        rng = uniform_range(("As", "Ks"), self.NUTS_BOARD[:4])
        dist = SubgameResolver(iterations=800, seed=1, cap_pot=True).resolve(
            s, cards("As", "Ks"), rng)
        self.assertNotIn(DOUBLE_POT, dist)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)

    def test_uncapped_resolver_includes_overbet_action(self):
        from nlhe_engine import DOUBLE_POT
        s = self._ext_river(("As", "Ks"))
        rng = uniform_range(("As", "Ks"), self.NUTS_BOARD[:4])
        dist = SubgameResolver(iterations=800, seed=1, cap_pot=False).resolve(
            s, cards("As", "Ks"), rng)
        self.assertIn(DOUBLE_POT, dist)


if __name__ == "__main__":
    unittest.main(verbosity=2)
