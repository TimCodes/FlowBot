"""DeepStack-style continual river re-solving (Moravcik et al. 2017).

The decisive difference from safe_resolver.py, and the fix for the measured
input-fidelity failure: **the opponent's range is not an input.** DeepStack's
re-solve consumes exactly two things -- OUR range and the opponent's
counterfactual value vector -- and our triangulated failures (unsafe -1721,
safe 3k -2428, safe 15k -2307 mbb/hand on river hands, vs the blueprint's
own -601) all shared one poisoned input: an opponent range estimated as
blueprint reach over the abstract-mapped line, which is systematically wrong
about a real opponent that does not share our abstraction.

Construction per river decision:

  1. OUR range: blueprint reach over our own actions -- trustworthy, because
     we genuinely play the blueprint before the river. After each re-solved
     action of ours, the range is updated by Bayes with the strategy the
     solve itself produced (continual re-solving), so within-hand
     consistency is preserved.
  2. Opponent hands: ALL combos consistent with the board, dealt uniformly
     in the gadget (card-removal by rejection against our sampled hand).
     No reach model of the opponent anywhere.
  3. CBV(h): best-response value of h against our range playing the
     blueprint from this node (bucketed BR, reused from safe_resolver).
     The gadget's Terminate option pins every hand to at least this value,
     which is what bounds our exploitability.

Deviations from the paper, noted honestly: CBVs are recomputed at each
decision from a bucketed BR rather than carried exactly from the previous
solve; no leaf value network (river subgames are solved to the end of the
hand exactly); our-range posterior updates use the solved root strategy
only. River-only.
"""

from __future__ import annotations

import random

from holdem_engine import FULL_DECK
from safe_resolver import SafeRiverResolver, _OurBuckets, compute_cbvs


def uniform_opponent_combos(board, block=()):
    """All hole combos consistent with the board (and optional blockers),
    uniformly weighted. This replaces the reach-estimated opponent range."""
    blocked = set(board) | set(block)
    deck = [c for c in FULL_DECK if c not in blocked]
    combos = [(deck[i], deck[j]) for i in range(len(deck))
              for j in range(i + 1, len(deck))]
    w = 1.0 / len(combos)
    return {h: w for h in combos}


class DeepStackResolver(SafeRiverResolver):
    """Gadget re-solve with uniform opponent dealing + CBV constraints.

    Inherits the gadget CFR from SafeRiverResolver; only the inputs differ:
    callers pass OUR range and no opponent range. The opponent side is the
    full combo set, so the solve defends against every holding no worse
    than the blueprint would (per the CBVs), rather than best-responding
    to a fictional reach distribution.
    """

    def resolve_deepstack(self, shadow, our_hole, our_range, policy,
                          bucketer) -> dict[str, float]:
        opp_range = uniform_opponent_combos(shadow.board_revealed())
        return self.resolve(shadow, our_hole, opp_range, our_range,
                            policy, bucketer)

    def posterior_our_range(self, shadow, our_range, bucketer, action):
        """Bayes-update our range with the strategy the solve produced.

        For each hand g in our range, weight *= P(action | g's bucket) from
        the gadget's average strategy at the acted node. Hands whose bucket
        never takes `action` drop out; unvisited buckets keep their weight
        (uniform strategy assumption). Returns a normalized dict.
        """
        hist = shadow.history_str()
        legal = shadow.legal_actions()
        idx = legal.index(action)
        updated = {}
        for g, w in our_range.items():
            label = bucketer.label(g, shadow.board, 3)
            node = self.last_nodes.get(f"u{label}|{hist}")
            if node and sum(node[1]) > 0 and len(node[2]) == len(legal):
                p = node[1][idx] / sum(node[1])
            else:
                p = 1.0 / len(legal)
            if w * p > 0:
                updated[g] = w * p
        total = sum(updated.values())
        if total <= 0:
            return our_range
        return {g: w / total for g, w in updated.items()}
