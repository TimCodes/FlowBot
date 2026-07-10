"""Show the worst river-decision hands from a match log to see the mistake."""
import json
import sys

from nlhe_engine import ACTION_PROFILES
from slumbot_client import replay_abstract

log = sys.argv[1]
state_cls = ACTION_PROFILES[sys.argv[2] if len(sys.argv) > 2 else "ext"]

rows = []
for line in open(log):
    rec = json.loads(line)
    if not rec.get("action") or not rec.get("board") or len(rec["board"]) < 5:
        continue
    trace = []
    replay_abstract(rec["action"], rec["hole_cards"], rec["board"],
                    trace_out=trace, state_cls=state_cls)
    if any(seat == rec["client_pos"] and street == 3 for seat, street, *_ in trace):
        rows.append(rec)

rows.sort(key=lambda r: r["winnings"])
print(f"{len(rows)} river-decision hands; 12 worst:")
for r in rows[:12]:
    print(f"  w={r['winnings']:+7d} pos={r['client_pos']} "
          f"hole={r['hole_cards']} bot={r.get('bot_hole_cards')} "
          f"board={r['board']}\n            action={r['action']!r}")
