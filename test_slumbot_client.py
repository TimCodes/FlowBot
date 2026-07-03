"""Offline tests for the Slumbot protocol bridge (no network required).

Run with:  .venv\\Scripts\\python -m unittest test_slumbot_client -v

Covers the action-string parser against the documented examples from
slumbot.com/sample_api.py, the real->abstract bet classification, the
abstract->real incremental-action translation, and shadow-state replay.
"""

import unittest

from nlhe_engine import ALL_IN, CALL, FOLD, HALF_POT, POT, STACK
from slumbot_client import (abstract_to_incr, classify_bet, parse_action,
                            replay_abstract)

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
