"""Tests for the empirical Slumbot model.

Run with:  .venv\\Scripts\\python -m unittest test_slumbot_model -v
"""

import unittest

from treys import Card

from slumbot_model import (SlumbotModel, iter_slumbot_actions,
                           river_line_category, strength_percentile)


def cards(*names):
    return [Card.new(n) for n in names]


def record(action, client_pos=1, board=None, bot_hole=None, hand=1):
    return {"hand": hand, "action": action, "client_pos": client_pos,
            "hole_cards": ["As", "Ks"], "board": board or [],
            "bot_hole_cards": bot_hole, "winnings": 0}


class TestActionIteration(unittest.TestCase):
    def test_checked_down_hand_slumbot_bb(self):
        # client_pos=1 -> we are SB (pos 1), Slumbot is BB (pos 0).
        # b200c/kk/kk/kb200c: we open b200, bot calls; bot checks each street;
        # river: bot checks, we bet 200, bot calls.
        rec = record("b200c/kk/kk/kb200c")
        obs = list(iter_slumbot_actions(rec))
        acts = [(st, facing, cls) for st, pos, facing, cls, _ in obs]
        self.assertEqual(acts, [
            (0, "b0", "c"),      # preflop: facing 100 into 300 pot
            (1, "none", "k"),
            (2, "none", "k"),
            (3, "none", "k"),
            (3, "b0", "c"),      # river call: 200 into 600 -> 0.33 -> b0
        ])

    def test_bet_size_classification(self):
        # Slumbot as SB (client_pos=0 -> bot pos 1) opens 3x: raise-by 200
        # into pot-after-call 200 = a full pot-size raise -> frac 1.0 -> b2.
        rec = record("b300", client_pos=0)
        obs = list(iter_slumbot_actions(rec))
        self.assertEqual(len(obs), 1)
        st, pos, facing, cls, frac = obs[0]
        self.assertEqual((st, pos, facing, cls), (0, 1, "b0", "b2"))
        self.assertAlmostEqual(frac, 1.0)

    def test_all_in_classified(self):
        rec = record("b20000", client_pos=0)
        (_, _, _, cls, _), = iter_slumbot_actions(rec)
        self.assertEqual(cls, "allin")

    def test_fold_recorded(self):
        # We are BB (client_pos=0)? No: bot pos = 1 - client_pos. Use
        # client_pos=0 -> bot is SB (pos 1): bot folds to nothing impossible;
        # instead: we are SB? client_pos=1 -> bot BB: we open, bot folds.
        rec = record("b300f")
        obs = [(st, cls) for st, _, _, cls, _ in iter_slumbot_actions(rec)]
        self.assertEqual(obs, [(0, "f")])

    def test_our_actions_not_recorded(self):
        rec = record("b200c/kk/kk/kb200c")
        for _, pos, _, _, _ in iter_slumbot_actions(rec):
            self.assertEqual(pos, 0)  # only Slumbot's seat


class TestRiverLine(unittest.TestCase):
    BOARD = ["Qs", "Js", "Ts", "2c", "2d"]

    def test_checked(self):
        self.assertEqual(
            river_line_category(record("b200c/kk/kk/kk")), "checked")

    def test_called(self):
        self.assertEqual(
            river_line_category(record("b200c/kk/kk/kb400c")), "called")

    def test_bet_classes(self):
        # bot BB checks... use bot betting: client_pos=0, bot pos 1 acts
        # second postflop. b200c/kk/kk/kb300c: bot bets 300 into 400 = 0.75
        # -> b1 -> bet_big.
        self.assertEqual(
            river_line_category(record("b200c/kk/kk/kb300c", client_pos=0)),
            "bet_big")
        # 200 into 400 = 0.5 -> b1? 0.5 <= 0.75 -> b1 -> bet_big; use 100
        # into 400 = 0.25 -> b0 -> bet_small.
        self.assertEqual(
            river_line_category(record("b200c/kk/kk/kb100c", client_pos=0)),
            "bet_small")

    def test_no_river_returns_none(self):
        self.assertIsNone(river_line_category(record("b300f")))


class TestStrengthPercentile(unittest.TestCase):
    BOARD = cards("Qs", "Js", "Ts", "2c", "2d")

    def test_nuts_near_one(self):
        p = strength_percentile(cards("As", "Ks"), self.BOARD)
        self.assertGreater(p, 0.99)

    def test_air_near_zero(self):
        p = strength_percentile(cards("4h", "3h"), self.BOARD)
        self.assertLess(p, 0.15)

    def test_monotone(self):
        strong = strength_percentile(cards("Qd", "Qh"), self.BOARD)  # boat
        mid = strength_percentile(cards("Ad", "Qc"), self.BOARD)     # pair Q
        self.assertGreater(strong, mid)


class TestModelBuildAndQuery(unittest.TestCase):
    def test_counts_and_smoothing(self):
        m = SlumbotModel()
        for _ in range(10):
            m.add_hand(record("b300f"), with_strength=False)
        probs, n = m.action_dist(0, 0, "b1")
        self.assertEqual(n, 10)
        self.assertGreater(probs["f"], 0.5)  # 10 folds + smoothing
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=9)

    def test_strength_histogram_roundtrip(self):
        m = SlumbotModel()
        rec = record("b200c/kk/kk/kb300c", client_pos=0,
                     board=["Qs", "Js", "Ts", "2c", "2d"],
                     bot_hole=["Ad", "Kd"])
        m.add_hand(rec)
        self.assertEqual(m.showdowns, 1)
        hist, n = m.strength_hist("bet_big")
        self.assertEqual(n, 1)
        self.assertAlmostEqual(sum(hist), 1.0, places=9)

    def test_save_load(self):
        import os
        import tempfile
        m = SlumbotModel()
        m.add_hand(record("b300f"), with_strength=False)
        path = os.path.join(tempfile.gettempdir(), "sm_test.pkl")
        m.save(path)
        m2 = SlumbotModel.load(path)
        self.assertEqual(m2.hands, 1)
        self.assertEqual(dict(m2.action_counts[(0, 0, "b1")]), {"f": 1})
        os.remove(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
