"""Tests for safe (gadget-game) river re-solving.

Run with:  .venv\\Scripts\\python -m unittest test_safe_resolver -v

The headline test is the safety property itself: the opponent's bucketed
best-response value against the GADGET-derived strategy must not beat its
best-response value against the blueprint (the CBVs) by more than solver
noise -- that bound is exactly what "safe" means.
"""

import unittest

from treys import Card

from card_abstraction import EquityBucketer
from nlhe_engine import ALL_IN, CALL, FOLD, HALF_POT, NLHEState, POT
from river_resolver import opponent_range
from safe_resolver import SafeRiverResolver, _OurBuckets, compute_cbvs


def cards(*names):
    return tuple(Card.new(n) for n in names)


NUTS_BOARD = ("Qs", "Js", "Ts", "2c", "2d")


def river_state(hole, board=NUTS_BOARD):
    s = NLHEState((cards(*hole), cards(*hole)), cards(*board))
    for _ in range(6):
        s = s.apply(CALL)
    assert s.street == 3 and s.to_act == 1
    return s


def make_ranges(hole):
    # samples must be high enough that river buckets are clean: at 10 MC
    # samples weak hands land in the top bucket often enough to corrupt the
    # bucket-average strategy (measured: 8-11% spurious fold with the nuts;
    # 0.0% at 50 samples).
    bucketer = EquityBucketer(samples=50, seed=2)
    opp = opponent_range({}, bucketer, [], 0, cards(*hole), cards(*NUTS_BOARD))
    ours = opponent_range({}, bucketer, [], 1, (), cards(*NUTS_BOARD))
    return bucketer, opp, ours


class TestCBV(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shadow = river_state(("9h", "8h"))
        cls.bucketer, cls.opp_range, cls.our_range = make_ranges(("9h", "8h"))
        cls.buckets = _OurBuckets(cls.our_range, cls.shadow.board,
                                  cls.bucketer, {})
        cls.cbv, cls.groups = compute_cbvs(cls.shadow, cls.opp_range,
                                           cls.buckets, opp_seat=0)

    @staticmethod
    def _find(cbv, *names):
        want = set(cards(*names))
        return next(v for h, v in cbv.items() if set(h) == want)

    def test_cbv_monotone_in_hand_strength(self):
        # A royal-flush holding must be worth more than complete air.
        # (Range keys are deck-ordered tuples, so match by card set.)
        nuts = self._find(self.cbv, "As", "Ks")
        air = self._find(self.cbv, "4h", "3h")
        self.assertGreater(nuts, air + 50)

    def test_cbv_bounded_by_stakes(self):
        # BR can always fold (lose current contribution, 100 chips here) and
        # can never win more than the opponent could ever put in.
        for v in self.cbv.values():
            self.assertGreaterEqual(v, -101)
            self.assertLessEqual(v, 20000)

    def test_groups_cover_range(self):
        self.assertEqual(set(self.cbv), set(self.opp_range))


class TestSafeResolver(unittest.TestCase):
    def test_returns_normalized_distribution(self):
        shadow = river_state(("9h", "8h"))
        bucketer, opp, ours = make_ranges(("9h", "8h"))
        dist = SafeRiverResolver(iterations=600, seed=1).resolve(
            shadow, cards("9h", "8h"), opp, ours, {}, bucketer)
        self.assertEqual(set(dist), set(shadow.legal_actions()))
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)

    def test_nuts_facing_bet_never_folds_and_raises(self):
        shadow = river_state(("As", "Ks")).apply(HALF_POT)
        bucketer, opp, ours = make_ranges(("As", "Ks"))
        dist = SafeRiverResolver(iterations=4000, seed=1).resolve(
            shadow, cards("As", "Ks"), opp, ours, {}, bucketer)
        self.assertLess(dist.get(FOLD, 0.0), 0.02)
        raise_mass = sum(dist.get(a, 0.0) for a in (HALF_POT, POT, ALL_IN))
        self.assertGreater(raise_mass, 0.7)

    def test_safety_property_aggregate(self):
        """Range-weighted opp BR value vs the gadget strategy must not beat
        the blueprint CBVs by more than solver/bucketing noise."""
        shadow = river_state(("9h", "8h"))
        bucketer, opp, ours = make_ranges(("9h", "8h"))
        resolver = SafeRiverResolver(iterations=4000, seed=1)
        resolver.resolve(shadow, cards("9h", "8h"), opp, ours, {}, bucketer)

        blueprint_buckets = _OurBuckets(ours, shadow.board, bucketer, {})
        cbv_bp, _ = compute_cbvs(shadow, opp, blueprint_buckets, opp_seat=0)

        # Rebuild our strategy from the gadget's average policy and re-run
        # the same best-response machinery against it.
        synth = {}
        for key, (reg, strat, acts) in resolver.last_nodes.items():
            if key.startswith("u") and sum(strat) > 0:
                label_hist = key[1:]
                total = sum(strat)
                synth[label_hist] = [s / total for s in strat]
        resolved_buckets = _OurBuckets(ours, shadow.board, bucketer, synth)
        cbv_res, _ = compute_cbvs(shadow, opp, resolved_buckets, opp_seat=0)

        agg_bp = sum(opp[h] * cbv_bp[h] for h in opp)
        agg_res = sum(opp[h] * cbv_res[h] for h in opp)
        # Pot here is 200 chips; allow noise but a genuinely unsafe strategy
        # (like the overcalling one) hands the BR hundreds of chips.
        self.assertLessEqual(agg_res, agg_bp + 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
