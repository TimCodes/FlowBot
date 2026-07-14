"""Diagnose the live safe-resolver failure by replaying logged river spots.

For each logged hand where we faced a river decision, rebuild the exact
decision context (action-string prefix at our first river turn), run the
safe resolver as the client would, and report the health of its inputs and
outputs:

  * T-rate: how often the gadget opponent Terminates. Near 1.0 means the
    Follow subtree (and hence our strategy) got no training signal and the
    safety bound is vacuous.
  * root visit mass: strategy_sum accumulated at our actual root infoset.
  * CBV stats vs the pot: overestimated CBVs push the opponent toward T.
  * the resolver's distribution vs the blueprint's at the same infoset.

Usage:
    .venv\\Scripts\\python diag_safe.py --log-jsonl slumbot_10k_ext_safe.jsonl --spots 10
"""

from __future__ import annotations

import argparse
import json
import pickle

from treys import Card

from card_abstraction import EquityBucketer
from hulhe_mccfr import infoset_key
from nlhe_engine import ACTION_PROFILES
from river_resolver import opponent_range
from safe_resolver import SafeRiverResolver
from slumbot_client import parse_action, replay_abstract


def action_prefixes(action):
    """Yield every valid prefix of a Slumbot action string at token ends."""
    i, n = 0, len(action)
    while i < n:
        c = action[i]
        if c == "/":
            i += 1
        elif c == "b":
            j = i + 1
            while j < n and action[j].isdigit():
                j += 1
            i = j
        else:
            i += 1
        yield action[:i]


def first_river_decision_prefix(action, my_pos):
    for prefix in action_prefixes(action):
        parsed = parse_action(prefix)
        if "error" in parsed:
            return None
        if parsed["pos"] == my_pos and parsed["st"] == 3:
            return prefix
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-jsonl", required=True)
    ap.add_argument("--blueprint", default="hunl_blueprint_ext.pkl")
    ap.add_argument("--spots", type=int, default=10)
    ap.add_argument("--iters", type=int, default=3000)
    args = ap.parse_args()

    saved = pickle.load(open(args.blueprint, "rb"))
    policy = saved["policy"]
    bucketer = EquityBucketer(saved["buckets"], saved["samples"], 0,
                              mode=saved.get("mode", "ehs"))
    state_cls = ACTION_PROFILES[saved.get("actions", "std")]

    shown = 0
    for line in open(args.log_jsonl):
        if shown >= args.spots:
            break
        rec = json.loads(line)
        if not rec.get("action") or len(rec.get("board") or []) < 5:
            continue
        prefix = first_river_decision_prefix(rec["action"], rec["client_pos"])
        if prefix is None:
            continue
        board_at = rec["board"]  # full board is known once street 3 reached
        trace = []
        shadow = replay_abstract(prefix, rec["hole_cards"], board_at,
                                 trace_out=trace, state_cls=state_cls)
        if shadow is None or shadow.is_terminal() or shadow.street != 3:
            continue
        our_hole = tuple(Card.new(c) for c in rec["hole_cards"])
        opp = opponent_range(policy, bucketer, trace, 1 - shadow.to_act,
                             our_hole, shadow.board_revealed())
        ours = opponent_range(policy, bucketer, trace, shadow.to_act, (),
                              shadow.board_revealed())
        resolver = SafeRiverResolver(args.iters, seed=0)
        dist = resolver.resolve(shadow, our_hole, opp, ours, policy, bucketer)

        # gadget T-rate across opponent groups
        t_rates, follow_mass = [], 0.0
        root_mass = 0.0
        for key, (reg, strat, acts) in resolver.last_nodes.items():
            if key.startswith("gadget|") and sum(strat) > 0:
                t_rates.append(strat[0] / sum(strat))
        actual_label = bucketer.label(our_hole, shadow.board, 3)
        root = resolver.last_nodes.get(
            f"u{actual_label}|{shadow.history_str()}")
        if root:
            root_mass = sum(root[1])

        from safe_resolver import _OurBuckets, compute_cbvs
        ob = _OurBuckets(ours, shadow.board, bucketer, policy)
        cbv, _ = compute_cbvs(shadow, opp, ob, 1 - shadow.to_act)
        pot = sum(shadow.contrib)
        cbvs = list(cbv.values())
        bp = policy.get(infoset_key(shadow, bucketer))

        print(f"hand {rec['hand']}: pot={pot} hist={shadow.history_str()!r} "
              f"hole={rec['hole_cards']}")
        print(f"   T-rate mean={sum(t_rates)/max(len(t_rates),1):.2f} "
              f"min={min(t_rates, default=0):.2f} "
              f"max={max(t_rates, default=0):.2f}   "
              f"root visit mass={root_mass:.1f}")
        print(f"   CBV: mean={sum(cbvs)/len(cbvs):+.0f} "
              f"min={min(cbvs):+.0f} max={max(cbvs):+.0f} (pot {pot})")
        print(f"   resolver dist={ {a: round(p, 2) for a, p in dist.items()} }")
        print(f"   blueprint    ={bp and [round(p, 2) for p in bp]}")
        shown += 1


if __name__ == "__main__":
    main()
