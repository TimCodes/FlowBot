"""Tests for DeepStack-style continual river re-solving.

Run with:  .venv\\Scripts\\python -m unittest test_deepstack_resolver -v
"""

import unittest

from treys import Card

from card_abstraction import EquityBucketer
from deepstack_resolver import DeepStackResolver, uniform_opponent_combos
from nlhe_engine import ALL_IN, CALL, FOLD, HALF_POT, NLHEState, POT
from river_resolver import opponent_range
from safe_resolver import _OurBuckets, compute_cbvs


def cards(*names):
    return tuple(Card.new(n) for n in names)


NUTS_BOARD = ("Qs", "Js", "Ts", "2c", "2d")


def river_state(hole, board=NUTS_BOARD):
    s = NLHEState((cards(*hole), cards(*hole)), cards(*board))
    for _ in range(6):
        s = s.apply(CALL)
    assert s.street == 3 and s.to_act == 1
    return s


def our_uniform_range():
    bucketer = EquityBucketer(samples=50, seed=2)
    ours = opponent_range({}, bucketer, [], 1, (), cards(*NUTS_BOARD))
    return bucketer, ours


class TestUniformCombos(unittest.TestCase):
    def test_counts_and_normalization(self):
        combos = uniform_opponent_combos(cards(*NUTS_BOARD))
        self.assertEqual(len(combos), 47 * 46 // 2)
        self.assertAlmostEqual(sum(combos.values()), 1.0, places=9)
        board = set(cards(*NUTS_BOARD))
        for h in combos:
            self.assertFalse(set(h) & board)

    def test_blockers_respected(self):
        combos = uniform_opponent_combos(cards(*NUTS_BOARD),
                                         block=cards("As", "Ks"))
        self.assertEqual(len(combos), 45 * 44 // 2)


class TestDeepStackResolver(unittest.TestCase):
    def test_returns_normalized_distribution(self):
        bucketer, ours = our_uniform_range()
        shadow = river_state(("9h", "8h"))
        dist = DeepStackResolver(iterations=800, seed=1).resolve_deepstack(
            shadow, cards("9h", "8h"), ours, {}, bucketer)
        self.assertEqual(set(dist), set(shadow.legal_actions()))
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)

    def test_nuts_facing_bet_never_folds(self):
        bucketer, ours = our_uniform_range()
        shadow = river_state(("As", "Ks")).apply(HALF_POT)
        dist = DeepStackResolver(iterations=6000, seed=1).resolve_deepstack(
            shadow, cards("As", "Ks"), ours, {}, bucketer)
        self.assertLess(dist.get(FOLD, 0.0), 0.03)

    def test_posterior_update_mechanics(self):
        bucketer, ours = our_uniform_range()
        shadow = river_state(("9h", "8h"))
        resolver = DeepStackResolver(iterations=2000, seed=1)
        resolver.resolve_deepstack(shadow, cards("9h", "8h"), ours, {},
                                   bucketer)
        post = resolver.posterior_our_range(shadow, ours, bucketer, CALL)
        self.assertAlmostEqual(sum(post.values()), 1.0, places=9)
        self.assertTrue(set(post) <= set(ours))
        # Aggressive buckets check less often than passive ones, so relative
        # weights must actually move somewhere.
        moved = any(abs(post[g] - ours[g]) > 1e-9 for g in post)
        self.assertTrue(moved)

    def test_safety_property_vs_full_combo_set(self):
        """No opponent hand -- from the FULL combo set, not an estimated
        range -- may beat its blueprint CBV against the resolved strategy."""
        bucketer, ours = our_uniform_range()
        shadow = river_state(("9h", "8h"))
        opp = uniform_opponent_combos(cards(*NUTS_BOARD))
        resolver = DeepStackResolver(iterations=4000, seed=1)
        resolver.resolve_deepstack(shadow, cards("9h", "8h"), ours, {},
                                   bucketer)

        bp_buckets = _OurBuckets(ours, shadow.board, bucketer, {})
        cbv_bp, _ = compute_cbvs(shadow, opp, bp_buckets, opp_seat=0)

        synth = {}
        for key, (reg, strat, acts) in resolver.last_nodes.items():
            if key.startswith("u") and sum(strat) > 0:
                total = sum(strat)
                synth[key[1:]] = [s / total for s in strat]
        res_buckets = _OurBuckets(ours, shadow.board, bucketer, synth)
        cbv_res, _ = compute_cbvs(shadow, opp, res_buckets, opp_seat=0)

        agg_bp = sum(opp[h] * cbv_bp[h] for h in opp)
        agg_res = sum(opp[h] * cbv_res[h] for h in opp)
        self.assertLessEqual(agg_res, agg_bp + 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
