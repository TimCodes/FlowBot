"""Heads-up limit Texas Hold'em (HULHE) game engine.

Rules (standard ACPC-style limit hold'em, heads-up):
  * Blinds 1/2. Player 0 is the small blind / button, player 1 the big blind.
  * Four streets. Bet size is 2 (small bet) preflop and on the flop, 4 (big
    bet) on the turn and river.
  * At most MAX_BETS=4 bets per street (one bet + three raises). Preflop the
    big blind counts as the first bet.
  * Player 0 acts first preflop; player 1 acts first on every later street.
  * Fold is only legal when facing a bet (folding instead of checking is
    dominated, so it is pruned from the tree, as is standard).

States are immutable from the caller's perspective: `apply(action)` returns a
new state. Hand evaluation uses treys (lower rank = stronger hand).
"""

from __future__ import annotations

from treys import Card, Evaluator

RANKS = "23456789TJQKA"
SUITS = "shdc"
FULL_DECK = tuple(Card.new(r + s) for r in RANKS for s in SUITS)

_evaluator = Evaluator()

FOLD, CALL, RAISE = "f", "c", "r"
BOARD_N = (0, 3, 4, 5, 5)  # revealed board cards per street; index 4 = showdown
BET_SIZE = (2, 2, 4, 4)
MAX_BETS = 4
SMALL_BLIND, BIG_BLIND = 1, 2
SHOWDOWN = 4


class HULHEState:
    __slots__ = ("holes", "board", "street", "hists", "bets", "acts",
                 "contrib", "to_act", "folded")

    def __init__(self, holes, board):
        self.holes = holes            # ((c, c), (c, c)) treys card ints
        self.board = tuple(board)     # 5 cards, revealed per street
        self.street = 0
        self.hists = [""]             # betting string per street
        self.bets = 1                 # the big blind is the first preflop bet
        self.acts = 0                 # actions taken this street
        self.contrib = [SMALL_BLIND, BIG_BLIND]
        self.to_act = 0
        self.folded = None

    def _clone(self) -> "HULHEState":
        s = object.__new__(HULHEState)
        s.holes = self.holes
        s.board = self.board
        s.street = self.street
        s.hists = list(self.hists)
        s.bets = self.bets
        s.acts = self.acts
        s.contrib = list(self.contrib)
        s.to_act = self.to_act
        s.folded = self.folded
        return s

    def board_revealed(self):
        return self.board[:BOARD_N[self.street]]

    def history_str(self) -> str:
        return "/".join(self.hists)

    def is_terminal(self) -> bool:
        return self.folded is not None or self.street == SHOWDOWN

    def legal_actions(self):
        me, opp = self.to_act, 1 - self.to_act
        actions = []
        if self.contrib[opp] > self.contrib[me]:
            actions.append(FOLD)
        actions.append(CALL)
        if self.bets < MAX_BETS:
            actions.append(RAISE)
        return actions

    def apply(self, action: str) -> "HULHEState":
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
            # Any call/check as the second-or-later action closes the street.
            # (The first action never closes it: an open limp leaves the big
            # blind an option; a first check leaves the opponent to act.)
            if s.acts >= 2:
                s._advance_street()
                return s
        elif action == RAISE:
            s.contrib[me] = s.contrib[opp] + BET_SIZE[s.street]
            s.bets += 1
        s.to_act = opp
        return s

    def _advance_street(self):
        self.street += 1
        if self.street < SHOWDOWN:
            self.hists.append("")
            self.bets = 0
            self.acts = 0
            self.to_act = 1  # big blind acts first postflop

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
        """Chips won (negative = lost) by `player` at a terminal state."""
        assert self.is_terminal()
        p0 = self._payoff0()
        return p0 if player == 0 else -p0
