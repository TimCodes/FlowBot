"""Card abstraction for HULHE: lossless preflop classes + equity buckets.

Preflop, the 1326 hole-card combos collapse losslessly (by suit isomorphism)
into the familiar 169 classes ("AKs", "T9o", "77", ...).

Postflop, hands are bucketed by Monte Carlo expected hand strength (EHS):
equity of (hole, board) against a uniform-random opponent hand over random
runouts, mapped into `num_buckets` fixed-width bins per street. Results are
cached per canonical (hole, board).

Notes for later rungs:
  * The literature prefers E[HS^2] or full equity-distribution clustering
    (potential-aware k-means) over plain EHS -- EHS conflates made hands with
    draws of equal mean equity. EHS is the honest baseline to beat.
  * Bucketing only the *current* street gives an imperfect-recall abstraction
    (the agent forgets which class its hand was in earlier). CFR's convergence
    guarantee technically requires perfect recall; imperfect-recall bucketing
    is standard practice and works well empirically, but flag it in writeups.
"""

from __future__ import annotations

import random
from collections import OrderedDict

from treys import Card

from holdem_engine import FULL_DECK, RANKS, _evaluator


def preflop_class(hole) -> str:
    r1, r2 = Card.get_rank_int(hole[0]), Card.get_rank_int(hole[1])
    hi, lo = max(r1, r2), min(r1, r2)
    if hi == lo:
        return RANKS[hi] * 2
    suited = Card.get_suit_int(hole[0]) == Card.get_suit_int(hole[1])
    return RANKS[hi] + RANKS[lo] + ("s" if suited else "o")


class EquityBucketer:
    def __init__(self, num_buckets: int = 8, samples: int = 50, seed: int = 0,
                 cache_limit: int = 2_000_000):
        self.num_buckets = num_buckets
        self.samples = samples
        self.rng = random.Random(seed)
        # OrderedDict so full-cache eviction is O(1) popitem(last=False).
        # A plain dict with pop(next(iter(...))) degrades badly: dict
        # iteration scans past deletion tombstones, so each eviction gets
        # slower the longer the cache has been at its limit.
        self.cache: OrderedDict[tuple, float] = OrderedDict()
        self.cache_limit = cache_limit  # bounds memory on multi-hour runs

    def hand_strength(self, hole, board) -> float:
        """Monte Carlo equity of `hole` on `board` vs a random opponent hand."""
        key = (tuple(sorted(hole)), tuple(sorted(board)))
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        blocked = set(hole) | set(board)
        remaining = [c for c in FULL_DECK if c not in blocked]
        need = 5 - len(board)
        hole_l, board_l = list(hole), list(board)
        score = 0.0
        for _ in range(self.samples):
            draw = self.rng.sample(remaining, 2 + need)
            runout = board_l + draw[2:]
            mine = _evaluator.evaluate(hole_l, runout)
            theirs = _evaluator.evaluate(draw[:2], runout)
            if mine < theirs:
                score += 1.0
            elif mine == theirs:
                score += 0.5
        hs = score / self.samples
        if len(self.cache) >= self.cache_limit:
            self.cache.popitem(last=False)  # O(1) FIFO eviction
        self.cache[key] = hs
        return hs

    def label(self, hole, board, street: int) -> str:
        """Abstraction label for an infoset: preflop class or street:bucket."""
        if street == 0:
            return preflop_class(hole)
        hs = self.hand_strength(hole, board)
        bucket = min(self.num_buckets - 1, int(hs * self.num_buckets))
        return f"{street}:{bucket}"
