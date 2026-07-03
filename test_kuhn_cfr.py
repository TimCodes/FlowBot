"""Verify CFR convergence against the analytical Nash equilibrium of Kuhn poker.

Run with:  python -m unittest test_kuhn_cfr -v   (or pytest)

The equilibrium is a one-parameter family (alpha in [0, 1/3]):
  P0: bet J with alpha, Q never, K with 3*alpha; after check-bet fold J,
      call Q with alpha + 1/3, call K.
  P1: facing a bet fold J, call Q with 1/3, call K;
      after a check bet J with 1/3, check Q, bet K.
Game value to P0 is -1/18. Tests check the alpha-invariant quantities plus
the family relations, and that NashConv (exploitability) approaches zero.
"""

import unittest

from kuhn_cfr import BET, PASS, KuhnCFRTrainer, nash_conv

GAME_VALUE = -1 / 18


class TestVanillaCFR(unittest.TestCase):
    ITERATIONS = 200_000

    @classmethod
    def setUpClass(cls):
        trainer = KuhnCFRTrainer(plus=False, seed=42)
        cls.value = trainer.train(cls.ITERATIONS)
        cls.avg = trainer.average_strategies()

    def test_game_value(self):
        self.assertAlmostEqual(self.value, GAME_VALUE, delta=0.01)

    def test_low_exploitability(self):
        self.assertLess(nash_conv(self.avg), 0.02)

    # --- Player 1 responses (alpha-invariant) ---

    def test_p1_folds_jack_to_bet(self):
        self.assertLess(self.avg["1b"][BET], 0.05)

    def test_p1_calls_queen_one_third(self):
        self.assertAlmostEqual(self.avg["2b"][BET], 1 / 3, delta=0.05)

    def test_p1_always_calls_king(self):
        self.assertGreater(self.avg["3b"][BET], 0.95)

    def test_p1_bluffs_jack_one_third_after_check(self):
        self.assertAlmostEqual(self.avg["1p"][BET], 1 / 3, delta=0.05)

    def test_p1_checks_queen_back(self):
        self.assertLess(self.avg["2p"][BET], 0.05)

    def test_p1_always_bets_king_after_check(self):
        self.assertGreater(self.avg["3p"][BET], 0.95)

    # --- Player 0 strategy (parameterized by alpha) ---

    def test_p0_never_opens_queen(self):
        self.assertLess(self.avg["2"][BET], 0.05)

    def test_p0_alpha_in_valid_range(self):
        alpha = self.avg["1"][BET]
        self.assertGreaterEqual(alpha, -1e-9)
        self.assertLessEqual(alpha, 1 / 3 + 0.05)

    def test_p0_king_bet_is_three_alpha(self):
        alpha = self.avg["1"][BET]
        self.assertAlmostEqual(self.avg["3"][BET], 3 * alpha, delta=0.06)

    def test_p0_queen_call_is_alpha_plus_third(self):
        alpha = self.avg["1"][BET]
        self.assertAlmostEqual(self.avg["2pb"][BET], alpha + 1 / 3, delta=0.06)

    def test_p0_folds_jack_to_raise(self):
        self.assertLess(self.avg["1pb"][BET], 0.05)

    def test_p0_always_calls_king_raise(self):
        self.assertGreater(self.avg["3pb"][BET], 0.95)


class TestCFRPlus(unittest.TestCase):
    """CFR+ should reach comparable exploitability in far fewer iterations."""

    ITERATIONS = 50_000

    @classmethod
    def setUpClass(cls):
        trainer = KuhnCFRTrainer(plus=True, seed=7)
        cls.value = trainer.train(cls.ITERATIONS)
        cls.avg = trainer.average_strategies()

    def test_game_value(self):
        self.assertAlmostEqual(self.value, GAME_VALUE, delta=0.02)

    def test_low_exploitability(self):
        self.assertLess(nash_conv(self.avg), 0.03)

    def test_p1_calls_queen_one_third(self):
        self.assertAlmostEqual(self.avg["2b"][BET], 1 / 3, delta=0.06)


class TestBestResponse(unittest.TestCase):
    """Sanity-check the best-response oracle itself."""

    def test_uniform_strategy_is_exploitable(self):
        # An empty dict means every infoset plays 50/50 -> clearly exploitable.
        self.assertGreater(nash_conv({}), 0.2)

    def test_nash_conv_nonnegative(self):
        trainer = KuhnCFRTrainer(seed=1)
        trainer.train(1_000)
        self.assertGreaterEqual(nash_conv(trainer.average_strategies()), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
