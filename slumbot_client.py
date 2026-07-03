"""Rung 4b: play the HUNL blueprint against Slumbot over its HTTP API.

Slumbot (slumbot.com) is the standard public HUNL benchmark: blinds 50/100,
200 BB stacks, reset every hand. Protocol (see slumbot.com/sample_api.py):

  * POST /slumbot/api/new_hand {token?} and /slumbot/api/act {token, incr}.
  * client_pos 0 = big blind (acts second preflop, first postflop), 1 = SB.
  * Action string: 'k' check, 'c' call, 'f' fold, 'bX' bet where X is the
    total chips that player has put in *on that street*; '/' separates
    streets; an all-in call may be followed by '///' filler.

Bridging real no-limit action onto the abstract blueprint (f/c/h/p/a):

  * Incoming bets are mapped to the nearest abstract size by the ratio
    raise_by / pot_after_call (h=0.5, p=1.0, all-in), thresholds 0.75/1.5.
    This is the naive scheme; the principled upgrade is pseudo-harmonic
    action translation (Ganzfried & Sandholm 2013).
  * A shadow abstract NLHEState is replayed from scratch on every decision
    (opponent hole cards and undealt board are dummy-filled -- the infoset
    key never reads them), then PolicyAgent picks the abstract action, which
    is converted back to a legal real bet.

Usage:
    .venv\\Scripts\\python slumbot_client.py --hands 100
    .venv\\Scripts\\python slumbot_client.py --hands 100 --blueprint hunl_blueprint.pkl
"""

from __future__ import annotations

import argparse
import pickle
import random
import time

import requests
from treys import Card

from card_abstraction import EquityBucketer
from hulhe_mccfr import PolicyAgent
from nlhe_engine import (ALL_IN, BIG_BLIND, CALL, FOLD, HALF_POT, NLHEState,
                         POT, SMALL_BLIND, STACK)

HOST = "slumbot.com"
NUM_STREETS = 4


# --- Action-string parsing (ported from slumbot.com/sample_api.py, with
# --- per-position contribution tracking added) -------------------------------

def parse_action(action: str) -> dict:
    """Parse a Slumbot action string.

    Positions use Slumbot numbering: 0 = big blind, 1 = small blind.
    Returns pos == -1 when the hand's betting is over. street_contrib and
    total_contrib are dicts keyed by position.
    """
    st = 0
    street_last_bet_to = BIG_BLIND
    last_bet_size = BIG_BLIND - SMALL_BLIND
    last_bettor = 0
    pos = 1
    street_contrib = {0: BIG_BLIND, 1: SMALL_BLIND}
    total_contrib = {0: BIG_BLIND, 1: SMALL_BLIND}
    check_or_call_ends_street = False
    i, sz = 0, len(action)

    def result(p):
        return {"st": st, "pos": p, "street_last_bet_to": street_last_bet_to,
                "last_bet_size": last_bet_size, "last_bettor": last_bettor,
                "street_contrib": street_contrib,
                "total_contrib": total_contrib}

    while i < sz:
        if st >= NUM_STREETS:
            return {"error": "unexpected action after showdown"}
        c = action[i]
        i += 1
        if c in ("k", "c"):
            if c == "k" and last_bet_size > 0:
                return {"error": "illegal check"}
            if c == "c":
                if last_bet_size == 0:
                    return {"error": "illegal call"}
                street_contrib[pos] = street_last_bet_to
                total_contrib[pos] = total_contrib[1 - pos]
                if total_contrib[pos] == STACK:  # call of an all-in
                    while i < sz and action[i] == "/":
                        i += 1
                    st = NUM_STREETS - 1
                    last_bet_size = 0
                    last_bettor = -1
                    return result(-1)
                last_bet_size = 0
                last_bettor = -1
            if check_or_call_ends_street:
                if st < NUM_STREETS - 1 and i < sz:
                    if action[i] != "/":
                        return {"error": "missing slash"}
                    i += 1
                if st == NUM_STREETS - 1:
                    return result(-1)  # showdown
                st += 1
                pos = 0  # big blind acts first postflop
                street_last_bet_to = 0
                street_contrib = {0: 0, 1: 0}
                check_or_call_ends_street = False
            else:
                pos = 1 - pos
                check_or_call_ends_street = True
        elif c == "f":
            if last_bet_size == 0:
                return {"error": "illegal fold"}
            return result(-1)
        elif c == "b":
            j = i
            while i < sz and action[i].isdigit():
                i += 1
            if i == j:
                return {"error": "missing bet size"}
            bet_to = int(action[j:i])
            raise_by = bet_to - street_last_bet_to
            total_contrib[pos] += bet_to - street_contrib[pos]
            street_contrib[pos] = bet_to
            last_bet_size = raise_by
            street_last_bet_to = bet_to
            last_bettor = pos
            pos = 1 - pos
            check_or_call_ends_street = True
        else:
            return {"error": f"unexpected character {c!r}"}
    return result(pos)


# --- Real <-> abstract action translation ------------------------------------

def classify_bet(raise_by: int, pot_after_call: int, bettor_total: int) -> str:
    """Map a real bet to the nearest abstract action by pot ratio."""
    if bettor_total >= STACK:
        return ALL_IN
    ratio = raise_by / max(pot_after_call, 1)
    if ratio < 0.75:
        return HALF_POT
    if ratio < 1.5:
        return POT
    return ALL_IN


def replay_abstract(action: str, hole_cards, board_cards) -> NLHEState | None:
    """Rebuild the shadow abstract state matching a real action string.

    Our seat in the shadow game is derived per real action from Slumbot
    positions; the shadow only needs OUR hole cards and the visible board --
    the opponent's hole and future board cards are dummy-filled.
    """
    seen = {Card.new(c) for c in hole_cards} | {Card.new(c) for c in board_cards}
    dummies = [c for c in _full_deck() if c not in seen][:7]
    hole = tuple(Card.new(c) for c in hole_cards)
    board = tuple(Card.new(c) for c in board_cards)
    board = board + tuple(dummies[2:2 + 5 - len(board)])

    # Track real contributions in parallel to compute bet ratios.
    st_contrib = {0: BIG_BLIND, 1: SMALL_BLIND}
    tot_contrib = {0: BIG_BLIND, 1: SMALL_BLIND}
    street_bet_to = BIG_BLIND
    pos = 1

    # Shadow state: seat 0 = small blind. We don't know which Slumbot pos we
    # are here; infoset_key only reads state.holes[state.to_act], so we give
    # BOTH shadow seats our hole cards. Only the acting player's view is read.
    shadow = NLHEState((hole, hole), board)
    i, sz = 0, len(action)
    while i < sz and not shadow.is_terminal():
        c = action[i]
        i += 1
        if c == "/":
            continue
        if c in ("k", "c"):
            abstract = CALL
            if c == "c":
                st_contrib[pos] = street_bet_to
                tot_contrib[pos] = tot_contrib[1 - pos]
        elif c == "f":
            abstract = FOLD
        elif c == "b":
            j = i
            while i < sz and action[i].isdigit():
                i += 1
            bet_to = int(action[j:i])
            raise_by = bet_to - street_bet_to
            to_call = street_bet_to - st_contrib[pos]
            pot_after_call = tot_contrib[0] + tot_contrib[1] + to_call
            tot_contrib[pos] += bet_to - st_contrib[pos]
            st_contrib[pos] = bet_to
            street_bet_to = bet_to
            abstract = classify_bet(raise_by, pot_after_call, tot_contrib[pos])
        else:
            return None
        # Apply with legality fallback (abstraction may prune sizes).
        legal = shadow.legal_actions()
        for candidate in _fallback_chain(abstract):
            if candidate in legal:
                prev_street = shadow.street
                shadow = shadow.apply(candidate)
                break
        else:
            return None
        if c in ("k", "c") and shadow.street != prev_street:
            st_contrib = {0: 0, 1: 0}
            street_bet_to = 0
            pos = 0
            continue
        pos = 1 - pos
    return shadow


def _fallback_chain(abstract: str):
    chains = {FOLD: (FOLD, CALL), CALL: (CALL,),
              HALF_POT: (HALF_POT, POT, ALL_IN, CALL),
              POT: (POT, HALF_POT, ALL_IN, CALL),
              ALL_IN: (ALL_IN, POT, HALF_POT, CALL)}
    return chains[abstract]


def _full_deck():
    from holdem_engine import FULL_DECK
    return FULL_DECK


def abstract_to_incr(abstract: str, parsed: dict, my_pos: int) -> str:
    """Convert our abstract action into a legal Slumbot incremental action."""
    facing = parsed["last_bettor"] not in (-1, my_pos)
    if abstract == FOLD:
        return "f" if facing else "k"
    if abstract == CALL:
        return "c" if facing else "k"

    my_street = parsed["street_contrib"][my_pos]
    my_total = parsed["total_contrib"][my_pos]
    opp_total = parsed["total_contrib"][1 - my_pos]
    street_bet_to = parsed["street_last_bet_to"]
    to_call = street_bet_to - my_street
    pot_after_call = my_total + opp_total + to_call
    remaining = STACK - my_total

    if abstract == HALF_POT:
        raise_by = pot_after_call // 2
    elif abstract == POT:
        raise_by = pot_after_call
    else:
        raise_by = remaining  # all-in

    min_raise = max(parsed["last_bet_size"], BIG_BLIND)
    raise_by = max(raise_by, min_raise)
    if to_call + raise_by >= remaining:
        bet_to = my_street + remaining  # all-in
    else:
        bet_to = street_bet_to + raise_by
    if bet_to <= street_bet_to:  # cannot legally raise: call instead
        return "c" if facing else "k"
    return f"b{bet_to}"


# --- HTTP plumbing ------------------------------------------------------------

def _post(endpoint: str, data: dict, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            resp = requests.post(f"https://{HOST}/slumbot/api/{endpoint}",
                                 json=data, timeout=30)
            body = resp.json()
            if resp.status_code != 200 or "error_msg" in body:
                raise RuntimeError(f"{endpoint}: {resp.status_code} "
                                   f"{body.get('error_msg', body)}")
            return body
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"{endpoint} failed: {exc}") from exc
            time.sleep(2 * (attempt + 1))


class SlumbotSession:
    def __init__(self, agent: PolicyAgent, token: str | None = None,
                 verbose: bool = False):
        self.agent = agent
        self.token = token
        self.verbose = verbose

    def _refresh_token(self, response: dict):
        if response.get("token"):
            self.token = response["token"]

    def play_hand(self) -> int:
        r = _post("new_hand", {"token": self.token} if self.token else {})
        self._refresh_token(r)
        my_pos = r.get("client_pos")
        while "winnings" not in r or r.get("winnings") is None:
            action = r["action"]
            parsed = parse_action(action)
            if "error" in parsed:
                raise RuntimeError(f"parse error on {action!r}: {parsed['error']}")
            my_pos = r.get("client_pos", my_pos)
            shadow = replay_abstract(action, r["hole_cards"], r["board"])
            if shadow is None or shadow.is_terminal():
                incr = "c" if parsed["last_bettor"] not in (-1, my_pos) else "k"
            else:
                abstract = self.agent.act(shadow)
                incr = abstract_to_incr(abstract, parsed, my_pos)
            if self.verbose:
                print(f"  action={action!r} hole={r['hole_cards']} "
                      f"board={r['board']} -> {incr}")
            r = _post("act", {"token": self.token, "incr": incr})
            self._refresh_token(r)
        return r["winnings"]


def main():
    parser = argparse.ArgumentParser(description="Play the blueprint vs Slumbot")
    parser.add_argument("--hands", type=int, default=100)
    parser.add_argument("--blueprint", default="hunl_blueprint.pkl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.blueprint, "rb") as f:
        saved = pickle.load(f)
    bucketer = EquityBucketer(saved["buckets"], saved["samples"], args.seed)
    agent = PolicyAgent(saved["policy"], bucketer, seed=args.seed)
    session = SlumbotSession(agent, verbose=args.verbose)

    print(f"Playing {args.hands} hands vs Slumbot "
          f"(blueprint: {saved['iterations']} MCCFR iterations)", flush=True)
    total = 0
    for h in range(1, args.hands + 1):
        total += session.play_hand()
        if h % 10 == 0 or h == args.hands:
            mbb_per_hand = total / BIG_BLIND / h * 1000
            print(f"hand {h:>5}: total {total:>+8} chips, "
                  f"{mbb_per_hand:>+9.1f} mbb/hand", flush=True)
    print(f"\nFinal: {total:+} chips over {args.hands} hands = "
          f"{total / BIG_BLIND / args.hands * 1000:+.1f} mbb/hand "
          f"(Slumbot baseline: random ~ -50000, always-call ~ -3000, "
          f"strong bots > 0)", flush=True)


if __name__ == "__main__":
    main()
