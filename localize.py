"""Localize the ext+resolve loss: split hands by whether WE faced a river
decision (where the resolver would fire) and compare mean win rate.

If resolve-fired hands are far more negative than the rest -- and the rest
match the ext-only baseline -- the resolver is the culprit. If the loss is
spread evenly, it is not the river at all.
"""
import json
import math
import sys

from treys import Card

from nlhe_engine import ACTION_PROFILES
import pickle
from slumbot_client import replay_abstract

log = sys.argv[1]
actions = sys.argv[2] if len(sys.argv) > 2 else "ext"
state_cls = ACTION_PROFILES[actions]

fired, notfired = [], []
for line in open(log):
    rec = json.loads(line)
    w = rec["winnings"]
    our_seat = rec.get("client_pos")
    action = rec.get("action")
    river_decision = False
    if action and rec.get("board") and len(rec["board"]) == 5:
        trace = []
        replay_abstract(action, rec["hole_cards"], rec["board"],
                        trace_out=trace, state_cls=state_cls)
        river_decision = any(seat == our_seat and street == 3
                             for seat, street, *_ in trace)
    (fired if river_decision else notfired).append(w)


def summ(xs, name):
    if not xs:
        print(f"{name}: (none)")
        return
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0
    se = sd / math.sqrt(n)
    print(f"{name}: n={n:5d}  {m/100*1000:+8.1f} +/- {se/100*1000:5.1f} mbb/hand")


print(f"log={log} profile={actions}")
summ(fired + notfired, "all hands          ")
summ(fired, "river-decision hands")
summ(notfired, "no river decision   ")
