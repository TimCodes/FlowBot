"""Offline tests for the Slumbot protocol bridge (no network required).

Run with:  .venv\\Scripts\\python -m unittest test_slumbot_client -v

Covers the action-string parser against the documented examples from
slumbot.com/sample_api.py, the real->abstract bet classification, the
abstract->real incremental-action translation, and shadow-state replay.
"""

import unittest

from nlhe_engine import ALL_IN, CALL, FOLD, HALF_POT, POT, STACK
from slumbot_client import (abstract_to_incr, classify_bet, parse_action,
                            pseudo_harmonic_prob, replay_abstract)

HOLE = ["As", "Ks"]


class TestParseAction(unittest.TestCase):
    def test_empty_action_sb_to_act(self):
        p = parse_action("")
        self.assertEqual(p["pos"], 1)          # small blind acts first
        self.assertEqual(p["street_last_bet_to"], 100)
        self.assertEqual(p["last_bettor"], 0)  # big blind posted the "bet"
        self.assertEqual(p["total_contrib"], {0: 100, 1: 50})

    def test_documented_multistreet_example(self):
        p = parse_action("b200c/kk/kk/kb200")
        self.assertEqual(p["st"], 3)
        self.assertEqual(p["pos"], 0)          # big blind facing river bet
        self.assertEqual(p["street_last_bet_to"], 200)
        self.assertEqual(p["last_bettor"], 1)
        self.assertEqual(p["total_contrib"], {0: 200, 1: 400})

    def test_documented_all_in_example(self):
        p = parse_action("b20000c///")
        self.assertEqual(p["pos"], -1)         # betting over, run out board
        self.assertEqual(p["total_contrib"], {0: STACK, 1: STACK})

    def test_limp_check_advances_street(self):
        p = parse_action("ck")
        self.assertEqual(p["st"], 1)
        self.assertEqual(p["pos"], 0)          # big blind first postflop
        self.assertEqual(p["street_last_bet_to"], 0)

    def test_fold_ends_hand(self):
        self.assertEqual(parse_action("f")["pos"], -1)

    def test_illegal_check_facing_blind(self):
        self.assertIn("error", parse_action("k"))

    def test_pot_size_flop_bet_example(self):
        # From the docs: "b200c/kb400" is a pot-size flop bet.
        p = parse_action("b200c/kb400")
        self.assertEqual(p["st"], 1)
        self.assertEqual(p["pos"], 0)
        self.assertEqual(p["last_bet_size"], 400)
        self.assertEqual(p["total_contrib"], {0: 200, 1: 600})


class TestBetClassification(unittest.TestCase):
    def test_half_pot(self):
        self.assertEqual(classify_bet(100, 200, 300), HALF_POT)

    def test_pot(self):
        self.assertEqual(classify_bet(200, 200, 400), POT)

    def test_overbet_is_all_in(self):
        self.assertEqual(classify_bet(600, 200, 800), ALL_IN)

    def test_stack_commitment_is_all_in(self):
        self.assertEqual(classify_bet(100, 200, STACK), ALL_IN)


class TestAbstractToIncr(unittest.TestCase):
    def test_sb_preflop_pot_raise_is_b300(self):
        p = parse_action("")
        self.assertEqual(abstract_to_incr(POT, p, my_pos=1), "b300")

    def test_sb_preflop_all_in_is_b20000(self):
        p = parse_action("")
        self.assertEqual(abstract_to_incr(ALL_IN, p, my_pos=1), "b20000")

    def test_call_when_facing(self):
        p = parse_action("b300")
        self.assertEqual(abstract_to_incr(CALL, p, my_pos=0), "c")

    def test_check_when_not_facing(self):
        p = parse_action("c")   # SB limped, BB to act with option
        self.assertEqual(abstract_to_incr(CALL, p, my_pos=0), "k")
        # Fold degrades to check when not facing a bet.
        self.assertEqual(abstract_to_incr(FOLD, p, my_pos=0), "k")

    def test_flop_half_pot_bet_after_limped_pot(self):
        p = parse_action("ck/")
        self.assertEqual(abstract_to_incr(HALF_POT, p, my_pos=0), "b100")

    def test_second_call_with_no_bet_is_illegal(self):
        self.assertIn("error", parse_action("cc/"))

    def test_min_raise_enforced(self):
        # Facing b400 on the flop: half-pot would be small; must be >= 400 more.
        p = parse_action("b200c/b400")
        incr = abstract_to_incr(HALF_POT, p, my_pos=0)
        self.assertTrue(incr.startswith("b"))
        self.assertGreaterEqual(int(incr[1:]), 800)


class TestPseudoHarmonic(unittest.TestCase):
    def test_boundaries(self):
        self.assertAlmostEqual(pseudo_harmonic_prob(0.5, 1.0, 0.5), 1.0)
        self.assertAlmostEqual(pseudo_harmonic_prob(0.5, 1.0, 1.0), 0.0)

    def test_known_value_from_paper(self):
        # A=0.5, B=1, x=0.75 -> (0.25 * 1.5) / (0.5 * 1.75) = 3/7
        self.assertAlmostEqual(pseudo_harmonic_prob(0.5, 1.0, 0.75), 3 / 7)

    def test_monotone_decreasing_in_x(self):
        probs = [pseudo_harmonic_prob(0.5, 1.0, x)
                 for x in (0.55, 0.65, 0.75, 0.85, 0.95)]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_harmonic_mapping_is_deterministic_per_salt(self):
        args = dict(raise_by=150, pot_after_call=200, bettor_total=400,
                    harmonic=True)
        first = classify_bet(**args, salt=17)
        self.assertTrue(all(classify_bet(**args, salt=17) == first
                            for _ in range(20)))

    def test_harmonic_frequencies_match_formula(self):
        # x = 0.75 between half-pot and pot: P(half-pot) should be ~3/7.
        picks = [classify_bet(150, 200, 400, harmonic=True, salt=s)
                 for s in range(2000)]
        frac_half = picks.count(HALF_POT) / len(picks)
        self.assertAlmostEqual(frac_half, 3 / 7, delta=0.04)
        self.assertEqual(set(picks), {HALF_POT, POT})

    def test_harmonic_extremes_are_deterministic(self):
        # At or below the smallest size, and at all-in, no randomization.
        self.assertEqual(classify_bet(100, 200, 300, harmonic=True), HALF_POT)
        self.assertEqual(classify_bet(19900, 200, STACK, harmonic=True), ALL_IN)

    def test_naive_mode_unchanged(self):
        self.assertEqual(classify_bet(100, 200, 300), HALF_POT)
        self.assertEqual(classify_bet(200, 200, 400), POT)
        self.assertEqual(classify_bet(600, 200, 800), ALL_IN)

    def test_extended_ladder_maps_overbets_to_d(self):
        from nlhe_engine import DOUBLE_POT, NLHEStateX
        ladder = NLHEStateX.RAISE_LADDER
        # A 2x-pot bet lands exactly on 'd' regardless of salt.
        picks = {classify_bet(400, 200, 600, harmonic=True, salt=s,
                              ladder=ladder) for s in range(50)}
        self.assertEqual(picks, {DOUBLE_POT})
        # A 1.5x-pot bet mixes between pot and double-pot only.
        picks = {classify_bet(300, 200, 500, harmonic=True, salt=s,
                              ladder=ladder) for s in range(200)}
        self.assertEqual(picks, {POT, DOUBLE_POT})

    def test_extended_incr_translation(self):
        from nlhe_engine import DOUBLE_POT, NLHEStateX
        p = parse_action("")
        incr = abstract_to_incr(DOUBLE_POT, p, my_pos=1,
                                ladder=NLHEStateX.RAISE_LADDER)
        self.assertEqual(incr, "b500")  # pot-after-call 200, raise by 400

    def test_extended_fallback_chain_orders_by_size(self):
        from nlhe_engine import DOUBLE_POT, NLHEStateX
        from slumbot_client import _fallback_chain
        chain = _fallback_chain(DOUBLE_POT, NLHEStateX.RAISE_LADDER)
        self.assertEqual(chain[0], DOUBLE_POT)
        self.assertEqual(chain[-1], CALL)
        self.assertLess(chain.index(POT), chain.index(HALF_POT))


class TestAivatLite(unittest.TestCase):
    def test_allin_call_street_detection(self):
        self.assertEqual(parse_action("b20000c///")["allin_call_st"], 0)
        self.assertEqual(parse_action("b200c/b19800c//")["allin_call_st"], 1)
        self.assertIsNone(parse_action("b200c/kk/kk/kb200")["allin_call_st"])
        self.assertIsNone(parse_action("f")["allin_call_st"])

    def test_river_allin_has_no_board_luck(self):
        from aivat_report import allin_luck_chips
        record = {"action": "b200c/kk/kk/kb19800c", "winnings": 20000,
                  "hole_cards": ["As", "Ad"], "bot_hole_cards": ["7c", "2d"],
                  "board": ["Ks", "Qh", "Jd", "8c", "3h"]}
        self.assertEqual(allin_luck_chips(record), 0.0)

    def test_preflop_allin_suckout_is_negative_luck(self):
        from aivat_report import allin_luck_chips
        # AA loses to 72o's full house after a preflop all-in: the realized
        # -20000 is far below the ~+15200 equity expectation.
        record = {"action": "b20000c///", "winnings": -20000,
                  "hole_cards": ["As", "Ad"], "bot_hole_cards": ["7c", "2d"],
                  "board": ["7h", "7s", "2h", "9c", "9d"]}
        self.assertLess(allin_luck_chips(record), -30000)

    def test_folded_hand_has_no_allin_luck(self):
        from aivat_report import allin_luck_chips
        record = {"action": "b300f", "winnings": 100,
                  "hole_cards": ["As", "Ad"], "bot_hole_cards": None,
                  "board": []}
        self.assertEqual(allin_luck_chips(record), 0.0)


class TestReplayAbstract(unittest.TestCase):
    def test_preflop_pot_raise_maps_to_p(self):
        shadow = replay_abstract("b300", HOLE, [])
        self.assertEqual(shadow.history_str(), "p")
        self.assertEqual(shadow.street, 0)
        self.assertEqual(shadow.to_act, 1)

    def test_documented_flop_pot_bet_maps_cleanly(self):
        shadow = replay_abstract("b200c/kb400", HOLE, ["Qs", "Js", "2c"])
        self.assertEqual(shadow.history_str(), "hc/cp")
        self.assertEqual(shadow.street, 1)

    def test_all_in_call_reaches_showdown(self):
        shadow = replay_abstract("b20000c///", HOLE, [])
        self.assertTrue(shadow.is_terminal())

    def test_empty_action(self):
        shadow = replay_abstract("", HOLE, [])
        self.assertEqual(shadow.history_str(), "")
        self.assertEqual(shadow.to_act, 0)  # our engine: SB seat acts first

    def test_fold_terminal(self):
        shadow = replay_abstract("b300f", HOLE, [])
        self.assertTrue(shadow.is_terminal())


if __name__ == "__main__":
    unittest.main(verbosity=2)
