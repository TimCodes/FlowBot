"""Probe: for one logged ext river hand, show each opponent decision's
policy-key lookup (hit/miss, length match) to locate the range bug."""
import json
import pickle
import sys

from treys import Card

from card_abstraction import EquityBucketer
from holdem_engine import BOARD_N
from nlhe_engine import ACTION_PROFILES
from slumbot_client import replay_abstract

bp = sys.argv[1] if len(sys.argv) > 1 else "hunl_blueprint_ext.pkl"
log = sys.argv[2] if len(sys.argv) > 2 else "slumbot_10k_ext_resolve.jsonl"

saved = pickle.load(open(bp, "rb"))
policy = saved["policy"]
state_cls = ACTION_PROFILES[saved.get("actions", "std")]
bucketer = EquityBucketer(saved["buckets"], saved["samples"], 0,
                          mode=saved.get("mode", "ehs"))
print(f"profile: {state_cls.__name__}")
print(f"blueprint {bp}: {len(policy)} keys, sample keys:")
for k in list(policy)[:6]:
    print(f"   {k!r} -> len {len(policy[k])}")

shown = 0
for line in open(log):
    rec = json.loads(line)
    if not rec.get("board") or len(rec["board"]) < 5 or not rec.get("action"):
        continue
    trace = []
    replay_abstract(rec["action"], rec["hole_cards"], rec["board"],
                    trace_out=trace, state_cls=state_cls)
    our_seat = rec["client_pos"]
    opp_seat = 1 - our_seat
    board = tuple(Card.new(c) for c in rec["board"])
    # Use a concrete plausible opponent hole (first two unused cards).
    used = {Card.new(c) for c in rec["hole_cards"]} | set(board)
    from holdem_engine import FULL_DECK
    opp_hole = tuple(c for c in FULL_DECK if c not in used)[:2]
    print(f"\nhand {rec['hand']}: action={rec['action']!r} "
          f"our_seat={our_seat} opp_hole test")
    for seat, street, hist, legal, chosen in trace:
        if seat != opp_seat:
            continue
        label = bucketer.label(opp_hole, board[:BOARD_N[street]], street)
        key = f"{label}|{hist}"
        probs = policy.get(key)
        status = ("MISS" if probs is None
                  else f"hit len={len(probs)} vs legal={len(legal)}"
                       f"{' LEN-MISMATCH' if len(probs) != len(legal) else ''}")
        print(f"   st{street} hist={hist!r:20} chosen={chosen} "
              f"legal={legal} key={key!r} -> {status}")
    shown += 1
    if shown >= 4:
        break
