"""Six-max no-limit Texas Hold'em engine with action abstraction (rung 5.1).

Generalizes nlhe_engine.py to 3-6 players. Blinds 50/100, every stack 20,000
(200 BB), reset each hand. Seat 0 is the small blind, seat 1 the big blind,
seat n-1 the button. Seat 2 (UTG) acts first preflop; postflop, action starts
with the first live seat from the small blind onward. Heads-up has different
ordering conventions (button = SB acts first preflop, last postflop), so
n >= 3 is enforced -- use nlhe_engine.NLHEState for the 2-player game.

The abstract action set is the same as rung 4 (this *is* the game the
blueprint trains on):

    f  fold (only when facing a bet)
    c  check / call
    h  raise by half the pot (pot measured after the call)
    p  raise by the full pot
    a  all-in

Raise legality follows no-limit rules: a raise must be at least the size of
the last bet/raise this street (the big blind preflop, one big blind to open
postflop). Undersized abstract raises are dropped rather than bumped; raises
meeting or exceeding the stack collapse into 'a'. Raising is disabled when no
other player can respond. Once at most one live player is not all-in at the
end of a street, the hand fast-forwards to showdown.

Multiway deltas with no heads-up analogue:
  * `pending` tracks how many live, non-all-in players still owe an action
    this street; a raise resets it, so the big blind's preflop option and
    re-opened action fall out naturally.
  * Payoffs are a vector settled with layered side pots (`settle`), which
    also handles dead money from folders and odd-chip splits. With equal
    starting stacks side pots only arise from folds, but `settle` is fully
    general so the engine survives future per-seat stacks.
  * A short all-in raise re-opens betting here (real NL rules say it should
    not); with 200 BB equal stacks the distinction is unreachable in the
    abstract game, so the simplification is documented rather than modeled.

Same state API as nlhe_engine.NLHEState plus vector `payoffs()`; `utility(p)`
returns that player's net chips, so agents and match runners port unchanged.
"""

from __future__ import annotations

from holdem_engine import BOARD_N, SHOWDOWN, _evaluator

FOLD, CALL, HALF_POT, POT, ALL_IN = "f", "c", "h", "p", "a"

SMALL_BLIND, BIG_BLIND = 50, 100
STACK = 20_000
NUM_PLAYERS = 6


def settle(contribs, folded, hand_ranks):
    """Net payoff vector from per-seat contributions via layered side pots.

    contribs: chips each seat put in; folded: set of folded seats;
    hand_ranks: seat -> showdown rank (treys, lower is better), required for
    every non-folded seat. Odd chips go to the earliest eligible seat.
    """
    n = len(contribs)
    payoff = [-c for c in contribs]
    remaining = list(contribs)
    while any(r > 0 for r in remaining):
        layer = min(r for r in remaining if r > 0)
        pot, eligible = 0, []
        for s in range(n):
            take = min(remaining[s], layer)
            pot += take
            remaining[s] -= take
            if take and s not in folded:
                eligible.append(s)
        best = min(hand_ranks[s] for s in eligible)
        winners = [s for s in eligible if hand_ranks[s] == best]
        share, odd = divmod(pot, len(winners))
        for j, s in enumerate(sorted(winners)):
            payoff[s] += share + (1 if j < odd else 0)
    return payoff


class NLHE6State:
    __slots__ = ("n", "holes", "board", "street", "hists", "contrib",
                 "folded", "to_act", "last_raise", "pending")

    def __init__(self, holes, board):
        n = len(holes)
        assert n >= 3, "use nlhe_engine.NLHEState for heads-up"
        self.n = n
        self.holes = holes
        self.board = tuple(board)
        self.street = 0
        self.hists = [""]
        self.contrib = [0] * n
        self.contrib[0] = SMALL_BLIND
        self.contrib[1] = BIG_BLIND
        self.folded = [False] * n
        self.to_act = 2  # UTG
        self.last_raise = BIG_BLIND  # preflop min-raise is one big blind
        self.pending = n  # everyone owes an action, incl. the BB's option

    def _clone(self) -> "NLHE6State":
        s = object.__new__(NLHE6State)
        s.n = self.n
        s.holes = self.holes
        s.board = self.board
        s.street = self.street
        s.hists = list(self.hists)
        s.contrib = list(self.contrib)
        s.folded = list(self.folded)
        s.to_act = self.to_act
        s.last_raise = self.last_raise
        s.pending = self.pending
        return s

    def board_revealed(self):
        return self.board[:BOARD_N[self.street]]

    def history_str(self) -> str:
        return "/".join(self.hists)

    def live_count(self) -> int:
        return self.n - sum(self.folded)

    def _can_act(self, i: int) -> bool:
        return not self.folded[i] and self.contrib[i] < STACK

    def is_terminal(self) -> bool:
        return self.live_count() == 1 or self.street == SHOWDOWN

    def raise_to_amount(self, action: str) -> int:
        """Total contribution a raise action would put the actor at."""
        if action == ALL_IN:
            return STACK
        cur_bet = max(self.contrib)
        to_call = cur_bet - self.contrib[self.to_act]
        pot_after_call = sum(self.contrib) + to_call
        raise_by = pot_after_call // 2 if action == HALF_POT else pot_after_call
        return min(STACK, cur_bet + raise_by)

    def legal_actions(self):
        me = self.to_act
        cur_bet = max(self.contrib)
        facing = cur_bet > self.contrib[me]
        actions = [FOLD, CALL] if facing else [CALL]
        others_respond = any(i != me and self._can_act(i) for i in range(self.n))
        if not others_respond or self.contrib[me] >= STACK:
            return actions
        min_raise_to = cur_bet + max(self.last_raise, BIG_BLIND)
        seen = set()
        for a in (HALF_POT, POT, ALL_IN):
            amount = self.raise_to_amount(a)
            if amount >= STACK:
                a, amount = ALL_IN, STACK
            elif amount < min_raise_to:
                continue
            if amount not in seen:
                seen.add(amount)
                actions.append(a)
        return actions

    def apply(self, action: str) -> "NLHE6State":
        assert not self.is_terminal()
        s = self._clone()
        me = s.to_act
        s.hists[-1] += action
        if action == FOLD:
            s.folded[me] = True
            s.pending -= 1
            if s.live_count() == 1:
                return s
        elif action == CALL:
            s.contrib[me] = max(s.contrib)
            s.pending -= 1
        else:
            amount = self.raise_to_amount(action)
            s.last_raise = amount - max(s.contrib)
            s.contrib[me] = amount
            s.pending = sum(1 for i in range(s.n)
                            if i != me and s._can_act(i))
        if s.pending == 0:
            s._advance_street()
        else:
            s.to_act = s._next_actor(me)
        return s

    def _next_actor(self, after: int) -> int:
        i = (after + 1) % self.n
        while not self._can_act(i):
            i = (i + 1) % self.n
        return i

    def _advance_street(self):
        actors = [i for i in range(self.n) if self._can_act(i)]
        if len(actors) <= 1:
            self.street = SHOWDOWN  # no betting left: run out the board
            return
        self.street += 1
        if self.street < SHOWDOWN:
            self.hists.append("")
            self.last_raise = 0  # postflop opening bet must be >= BIG_BLIND
            self.pending = len(actors)
            self.to_act = self._next_actor(self.n - 1)  # first live from SB

    def payoffs(self):
        assert self.is_terminal()
        live = [i for i in range(self.n) if not self.folded[i]]
        if len(live) == 1:
            ranks = {live[0]: 0}
        else:
            board = list(self.board)
            ranks = {i: _evaluator.evaluate(list(self.holes[i]), board)
                     for i in live}
        return settle(self.contrib, {i for i, f in enumerate(self.folded) if f},
                      ranks)

    def utility(self, player: int) -> int:
        return self.payoffs()[player]
