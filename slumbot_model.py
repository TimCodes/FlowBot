"""Empirical Slumbot model estimated from logged match hands.

Two components with very different data economics, kept separate on purpose:

  * ACTION model -- P(action | situation) for Slumbot, where a situation is
    (street, slumbot position, facing-size bucket). Every action in every
    logged hand is an observation (~300k observations from ~80k hands), so
    these estimates are dense and trustworthy.
  * STRENGTH model -- P(hand-strength percentile | Slumbot's river line),
    observable only at showdowns. This sample is SELECTION-BIASED (hands
    Slumbot folded are never observed; calls are over-represented), so every
    histogram carries its raw count `n` and consumers are expected to
    confidence-weight (Data-Biased Response style) rather than trust sparse
    cells. The bias direction is at least conservative for the question that
    killed the re-solvers ("are big river bets value-heavy?"): bluffs that
    fold out our agent never reach showdown, so observed big-bet strength is
    an overestimate of value-weighting -- noted in the report.

Build:
    .venv\\Scripts\\python slumbot_model.py --build "*.jsonl" --save slumbot_model.pkl
Query (programmatic):
    model = SlumbotModel.load("slumbot_model.pkl")
    probs, n = model.action_dist(street=3, pos=0, facing="b2")
    hist, n = model.strength_hist("bet_big")
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
from collections import defaultdict
from itertools import combinations

from treys import Card

from holdem_engine import FULL_DECK, _evaluator
from diag_safe import action_prefixes
from slumbot_client import STACK, parse_action

FACING_BUCKETS = ((0.4, "b0"), (0.75, "b1"), (1.33, "b2"), (2.5, "b3"))
ACTIONS = ("f", "k", "c", "b0", "b1", "b2", "b3", "allin")
STRENGTH_BINS = 20  # percentile histogram resolution


def frac_bucket(frac):
    for cap, name in FACING_BUCKETS:
        if frac <= cap:
            return name
    return "b3"


def iter_slumbot_actions(record):
    """Yield (street, slumbot_pos, facing, action_class, raise_frac) for each
    Slumbot action in a logged hand, using consecutive validated prefixes of
    the real action string (the battle-tested parser does the state
    tracking)."""
    action = record.get("action")
    if not action:
        return
    slumbot_pos = 1 - record["client_pos"]
    prev = ""
    before = parse_action("")
    for prefix in action_prefixes(action):
        token = prefix[len(prev):]
        prev_state, prev = before, prefix
        before = parse_action(prefix)
        if "error" in before:
            return
        token = token.rstrip("/")
        if not token or "error" in prev_state:
            continue
        actor = prev_state["pos"]
        if actor != slumbot_pos or actor == -1:
            continue
        st = prev_state["st"]
        street_bet = prev_state["street_last_bet_to"]
        my_street = prev_state["street_contrib"][actor]
        pot_before = (prev_state["total_contrib"][0]
                      + prev_state["total_contrib"][1])
        to_call = street_bet - my_street
        facing = "none" if to_call <= 0 else frac_bucket(to_call / pot_before)
        if token == "f":
            yield st, slumbot_pos, facing, "f", None
        elif token == "k":
            yield st, slumbot_pos, facing, "k", None
        elif token == "c":
            yield st, slumbot_pos, facing, "c", None
        elif token.startswith("b"):
            bet_to = int(token[1:])
            raise_by = bet_to - street_bet
            pot_after_call = pot_before + to_call
            frac = raise_by / max(pot_after_call, 1)
            total_after = prev_state["total_contrib"][actor] + (bet_to - my_street)
            cls = "allin" if total_after >= STACK else frac_bucket(frac)
            yield st, slumbot_pos, facing, cls, frac


def river_line_category(record):
    """Slumbot's river behavior class for the strength model."""
    best = None
    for st, _, facing, cls, frac in iter_slumbot_actions(record):
        if st != 3:
            continue
        if cls == "allin":
            return "allin"
        if cls.startswith("b"):
            rank = {"b0": "bet_small", "b1": "bet_big",
                    "b2": "overbet", "b3": "overbet"}[cls]
            best = rank if best in (None, "checked", "called",
                                    "bet_small") else best
            if rank == "overbet":
                best = "overbet"
        elif cls == "c" and best in (None, "checked"):
            best = "called"
        elif cls == "k" and best is None:
            best = "checked"
    return best


def strength_percentile(bot_hole, board):
    """Fraction of possible opposing combos this hand beats on this board."""
    blocked = set(bot_hole) | set(board)
    deck = [c for c in FULL_DECK if c not in blocked]
    my_rank = _evaluator.evaluate(list(bot_hole), list(board))
    wins = ties = total = 0
    for a, b in combinations(deck, 2):
        r = _evaluator.evaluate([a, b], list(board))
        total += 1
        if my_rank < r:
            wins += 1
        elif my_rank == r:
            ties += 1
    return (wins + 0.5 * ties) / total


class SlumbotModel:
    def __init__(self):
        self.action_counts = defaultdict(lambda: defaultdict(int))
        self.strength_counts = defaultdict(lambda: [0] * STRENGTH_BINS)
        self.hands = 0
        self.showdowns = 0

    # -- build ---------------------------------------------------------------

    def add_hand(self, record, with_strength=True):
        self.hands += 1
        for st, pos, facing, cls, _ in iter_slumbot_actions(record):
            self.action_counts[(st, pos, facing)][cls] += 1
        if (with_strength and record.get("bot_hole_cards")
                and len(record.get("board") or []) == 5):
            cat = river_line_category(record)
            if cat is not None:
                hole = [Card.new(c) for c in record["bot_hole_cards"]]
                board = [Card.new(c) for c in record["board"]]
                p = strength_percentile(hole, board)
                b = min(STRENGTH_BINS - 1, int(p * STRENGTH_BINS))
                self.strength_counts[cat][b] += 1
                self.showdowns += 1

    # -- queries -------------------------------------------------------------

    def action_dist(self, street, pos, facing, smooth=1.0):
        counts = self.action_counts.get((street, pos, facing), {})
        n = sum(counts.values())
        legal = ([a for a in ACTIONS if a != "k"] if facing != "none"
                 else [a for a in ACTIONS if a not in ("f", "c")])
        probs = {a: (counts.get(a, 0) + smooth) / (n + smooth * len(legal))
                 for a in legal}
        return probs, n

    def strength_hist(self, category):
        hist = self.strength_counts.get(category)
        if hist is None:
            return None, 0
        n = sum(hist)
        return ([h / n for h in hist] if n else None), n

    # -- persistence ---------------------------------------------------------

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"action_counts": {k: dict(v) for k, v
                                           in self.action_counts.items()},
                         "strength_counts": dict(self.strength_counts),
                         "hands": self.hands,
                         "showdowns": self.showdowns}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        m = cls()
        for k, v in d["action_counts"].items():
            m.action_counts[k].update(v)
        for k, v in d["strength_counts"].items():
            m.strength_counts[k] = list(v)
        m.hands = d["hands"]
        m.showdowns = d["showdowns"]
        return m


def report(model: SlumbotModel):
    streets = ["preflop", "flop", "turn", "river"]
    print(f"\n=== Slumbot empirical model: {model.hands:,} hands, "
          f"{model.showdowns:,} scored showdowns ===")
    print("\n-- action tendencies (both positions pooled) --")
    for st in range(4):
        merged_none, merged_face = defaultdict(int), defaultdict(int)
        for pos in (0, 1):
            for a, c in model.action_counts.get((st, pos, "none"), {}).items():
                merged_none[a] += c
            for facing in ("b0", "b1", "b2", "b3", "allin"):
                for a, c in model.action_counts.get((st, pos, facing),
                                                    {}).items():
                    merged_face[a] += c
        n0, n1 = sum(merged_none.values()), sum(merged_face.values())
        if n0:
            bet = sum(c for a, c in merged_none.items()
                      if a.startswith("b") or a == "allin") / n0
            print(f"  {streets[st]:>7} unfaced (n={n0:6d}): bet {bet:5.1%}")
        if n1:
            fold = merged_face.get("f", 0) / n1
            raise_ = sum(c for a, c in merged_face.items()
                         if a.startswith("b") or a == "allin") / n1
            print(f"  {streets[st]:>7} facing  (n={n1:6d}): fold {fold:5.1%}  "
                  f"raise {raise_:5.1%}  call {1 - fold - raise_:5.1%}")
    print("\n-- river strength by Slumbot line (showdown-biased sample!) --")
    print("   NOTE: bluffs that folded us out never show down, so big-bet")
    print("   strength below OVERSTATES value-weighting; treat as upper bound.")
    for cat in ("checked", "called", "bet_small", "bet_big", "overbet",
                "allin"):
        hist, n = model.strength_hist(cat)
        if not hist:
            continue
        mean = sum((i + 0.5) / STRENGTH_BINS * h for i, h in enumerate(hist))
        top20 = sum(hist[int(0.8 * STRENGTH_BINS):])
        bottom30 = sum(hist[:int(0.3 * STRENGTH_BINS)])
        print(f"  {cat:>10} (n={n:5d}): mean pct {mean:5.1%}   "
              f"top-20% hands {top20:5.1%}   bottom-30% hands {bottom30:5.1%}")


def main():
    ap = argparse.ArgumentParser(description="Build empirical Slumbot model")
    ap.add_argument("--build", default="*.jsonl",
                    help="glob of match logs to ingest")
    ap.add_argument("--save", default="slumbot_model.pkl")
    ap.add_argument("--strength-sample", type=int, default=0,
                    help="cap scored showdowns (0 = all; scoring is ~10ms "
                         "per showdown)")
    args = ap.parse_args()

    model = SlumbotModel()
    files = sorted(glob.glob(args.build))
    print(f"ingesting {len(files)} logs: {', '.join(files)}")
    scored = 0
    for path in files:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                with_strength = (args.strength_sample == 0
                                 or scored < args.strength_sample)
                before = model.showdowns
                model.add_hand(rec, with_strength=with_strength)
                scored += model.showdowns - before
        print(f"  {path}: cumulative {model.hands:,} hands, "
              f"{model.showdowns:,} showdowns scored", flush=True)
    model.save(args.save)
    print(f"saved {args.save}")
    report(model)


if __name__ == "__main__":
    main()
