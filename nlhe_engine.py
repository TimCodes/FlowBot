"""Heads-up no-limit Texas Hold'em (HUNL) engine with action abstraction.

ACPC/Slumbot conventions: blinds 50/100, both stacks 20,000 (200 BB), reset
every hand. Player 0 is the small blind / button and acts first preflop;
player 1 acts first postflop.

The action set is abstracted inside the engine -- this *is* the abstract game
the blueprint is trained on (Libratus/Pluribus style):

    f  fold (only when facing a bet)
    c  check / call
    h  raise by half the pot (pot measured after the call)
    p  raise by the full pot
    a  all-in

Raise legality follows no-limit rules: a raise must be at least the size of
the last bet/raise this street (min-raise; the big blind preflop, one big
blind to open postflop). Undersized abstract raises are dropped from the
action set rather than bumped; raises that would meet or exceed the stack
collapse into 'a'. When a call closes the action with both players all-in,
the state fast-forwards to showdown.

Same state API as holdem_engine.HULHEState, so the MCCFR trainer, agents,
and match runner in hulhe_mccfr.py work on both games unchanged.
"""

from __future__ import annotations

from holdem_engine import BOARD_N, SHOWDOWN, _evaluator

FOLD, CALL, HALF_POT, POT, ALL_IN = "f", "c", "h", "p", "a"
DOUBLE_POT = "d"  # only in the extended profile (NLHEStateX)

SMALL_BLIND, BIG_BLIND = 50, 100
STACK = 20_000


class NLHEState:
    # (letter, pot fraction) raise sizes, ordered small to large; ALL_IN is
    # always appended. Subclasses override to change the action profile --
    # policies are only compatible with the profile they were trained on.
    RAISE_LADDER = ((HALF_POT, 0.5), (POT, 1.0))

    __slots__ = ("holes", "board", "street", "hists", "acts", "contrib",
                 "to_act", "folded", "last_raise")

    def __init__(self, holes, board):
        self.holes = holes
        self.board = tuple(board)
        self.street = 0
        self.hists = [""]
        self.acts = 0
        self.contrib = [SMALL_BLIND, BIG_BLIND]
        self.to_act = 0
        self.folded = None
        self.last_raise = BIG_BLIND  # preflop min-raise is one big blind

    def _clone(self) -> "NLHEState":
        s = object.__new__(type(self))
        s.holes = self.holes
        s.board = self.board
        s.street = self.street
        s.hists = list(self.hists)
        s.acts = self.acts
        s.contrib = list(self.contrib)
        s.to_act = self.to_act
        s.folded = self.folded
        s.last_raise = self.last_raise
        return s

    def board_revealed(self):
        return self.board[:BOARD_N[self.street]]

    def history_str(self) -> str:
        return "/".join(self.hists)

    def is_terminal(self) -> bool:
        return self.folded is not None or self.street == SHOWDOWN

    def raise_to_amount(self, action: str) -> int:
        """Total contribution a raise action would put the actor at."""
        if action == ALL_IN:
            return STACK
        me, opp = self.to_act, 1 - self.to_act
        to_call = self.contrib[opp] - self.contrib[me]
        pot_after_call = self.contrib[me] + self.contrib[opp] + to_call
        raise_by = int(pot_after_call * dict(self.RAISE_LADDER)[action])
        return min(STACK, self.contrib[opp] + raise_by)

    def legal_actions(self):
        me, opp = self.to_act, 1 - self.to_act
        facing = self.contrib[opp] > self.contrib[me]
        actions = [FOLD, CALL] if facing else [CALL]
        if self.contrib[opp] >= STACK or self.contrib[me] >= STACK:
            return actions  # facing (or committed) all-in: fold/call only
        min_raise_to = self.contrib[opp] + max(self.last_raise, BIG_BLIND)
        seen = set()
        ladder = tuple(letter for letter, _ in self.RAISE_LADDER) + (ALL_IN,)
        for a in ladder:
            amount = self.raise_to_amount(a)
            if amount >= STACK:
                a, amount = ALL_IN, STACK
            elif amount < min_raise_to:
                continue
            if amount not in seen:
                seen.add(amount)
                actions.append(a)
        return actions

    def apply(self, action: str) -> "NLHEState":
        assert not self.is_terminal()
        s = self._clone()
        me, opp = s.to_act, 1 - s.to_act
        s.hists[-1] += action
        s.acts += 1
        if action == FOLD:
            s.folded = me
            return s
        if action == CALL:
            s.contrib[me] = s.contrib[opp]
            if s.acts >= 2:
                s._advance_street()
                return s
        else:
            amount = self.raise_to_amount(action)
            s.last_raise = amount - s.contrib[opp]
            s.contrib[me] = amount
        s.to_act = opp
        return s

    def _advance_street(self):
        if self.contrib[0] >= STACK and self.contrib[1] >= STACK:
            self.street = SHOWDOWN  # both all-in: run out the board
            return
        self.street += 1
        if self.street < SHOWDOWN:
            self.hists.append("")
            self.acts = 0
            self.to_act = 1
            self.last_raise = 0  # postflop opening bet must be >= BIG_BLIND

    def _payoff0(self) -> int:
        if self.folded == 0:
            return -self.contrib[0]
        if self.folded == 1:
            return self.contrib[1]
        board = list(self.board)
        r0 = _evaluator.evaluate(list(self.holes[0]), board)
        r1 = _evaluator.evaluate(list(self.holes[1]), board)
        if r0 < r1:
            return self.contrib[1]
        if r1 < r0:
            return -self.contrib[0]
        return 0

    def utility(self, player: int = 0) -> int:
        assert self.is_terminal()
        p0 = self._payoff0()
        return p0 if player == 0 else -p0


class NLHEStateX(NLHEState):
    """Extended action profile: adds a 2x-pot overbet ('d').

    Policies trained on one profile are incompatible with the other (the
    legal-action lists differ), so blueprint pickles record which profile
    they were trained with ("actions": "std" | "ext").
    """

    RAISE_LADDER = ((HALF_POT, 0.5), (POT, 1.0), (DOUBLE_POT, 2.0))

    __slots__ = ()


ACTION_PROFILES = {"std": NLHEState, "ext": NLHEStateX}
