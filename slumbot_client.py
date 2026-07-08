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
import json
import os
import pickle
import random
import time

import requests
from treys import Card

from card_abstraction import EquityBucketer
from hulhe_mccfr import PolicyAgent
from hunl_blueprint import keep_system_awake
from nlhe_engine import (ACTION_PROFILES, ALL_IN, BIG_BLIND, CALL, FOLD,
                         HALF_POT, NLHEState, POT, SMALL_BLIND, STACK)

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
    allin_call_st = None  # street where an all-in bet got called, if any
    i, sz = 0, len(action)

    def result(p):
        return {"st": st, "pos": p, "street_last_bet_to": street_last_bet_to,
                "last_bet_size": last_bet_size, "last_bettor": last_bettor,
                "street_contrib": street_contrib,
                "total_contrib": total_contrib,
                "allin_call_st": allin_call_st}

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
                    allin_call_st = st
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

def pseudo_harmonic_prob(a: float, b: float, x: float) -> float:
    """P(map a bet of pot-fraction x to the smaller abstract size a).

    Pseudo-harmonic mapping, Ganzfried & Sandholm 2013 ("Action Translation
    in Extensive-Form Games with Large Action Spaces"). f_a(a)=1, f_a(b)=0,
    and the interpolation matches the Nash mapping of a simplified game;
    randomizing removes the exploitable hard threshold of nearest-size maps.
    """
    return (b - x) * (1 + a) / ((b - a) * (1 + x))


def classify_bet(raise_by: int, pot_after_call: int, bettor_total: int,
                 harmonic: bool = False, salt: int = 0,
                 ladder: tuple = NLHEState.RAISE_LADDER) -> str:
    """Map a real bet to an abstract action by pot ratio.

    harmonic=False with the standard 2-size ladder: legacy nearest-threshold
    mapping (baseline behaviour). Otherwise pseudo-harmonic randomized
    mapping over the (letter, fraction) `ladder` plus all-in; `salt` keeps
    the draw deterministic per bet so replays within a hand stay consistent.
    """
    if bettor_total >= STACK:
        return ALL_IN
    pot = max(pot_after_call, 1)
    x = raise_by / pot
    if not harmonic and ladder == NLHEState.RAISE_LADDER:
        if x < 0.75:
            return HALF_POT
        if x < 1.5:
            return POT
        return ALL_IN
    allin_x = (raise_by + STACK - bettor_total) / pot
    rungs = [r for r in ladder if r[1] < allin_x] + [(ALL_IN, allin_x)]
    if x <= rungs[0][1]:
        return rungs[0][0]
    if x >= allin_x:
        return ALL_IN
    for (low, a), (high, b) in zip(rungs, rungs[1:]):
        if a <= x < b:
            if b <= a:  # degenerate bracket (tiny remaining stack)
                return low
            p_low = pseudo_harmonic_prob(a, b, x)
            u = random.Random(f"{salt}:{x:.6f}").random()
            return low if u < p_low else high
    return ALL_IN


def replay_abstract(action: str, hole_cards, board_cards,
                    harmonic: bool = False,
                    trace_out: list | None = None,
                    state_cls: type = NLHEState) -> NLHEState | None:
    """Rebuild the shadow abstract state matching a real action string.

    Our seat in the shadow game is derived per real action from Slumbot
    positions; the shadow only needs OUR hole cards and the visible board --
    the opponent's hole and future board cards are dummy-filled.

    If `trace_out` is given, one record is appended per abstract decision:
    (seat, street, history_str_before, legal_actions, chosen_action) -- the
    raw material for Bayesian range estimation in river_resolver.py.
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
    shadow = state_cls((hole, hole), board)
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
            # salt = char index of this bet: the same bet maps to the same
            # abstract size every time the growing action string is replayed.
            abstract = classify_bet(raise_by, pot_after_call, tot_contrib[pos],
                                    harmonic=harmonic, salt=j,
                                    ladder=state_cls.RAISE_LADDER)
        else:
            return None
        # Apply with legality fallback (abstraction may prune sizes).
        legal = shadow.legal_actions()
        for candidate in _fallback_chain(abstract, state_cls.RAISE_LADDER):
            if candidate in legal:
                if trace_out is not None:
                    trace_out.append((shadow.to_act, shadow.street,
                                      shadow.history_str(), tuple(legal),
                                      candidate))
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


def _fallback_chain(abstract: str, ladder: tuple = NLHEState.RAISE_LADDER):
    """Substitution order when the desired action is pruned: nearest raise
    sizes first (all-in counts as the largest), then check/call."""
    if abstract == FOLD:
        return (FOLD, CALL)
    if abstract == CALL:
        return (CALL,)
    fracs = dict(ladder)
    fracs[ALL_IN] = 200.0  # effectively infinite pot fraction
    others = sorted((a for a in fracs if a != abstract),
                    key=lambda a: abs(fracs[a] - fracs[abstract]))
    return (abstract, *others, CALL)


def _full_deck():
    from holdem_engine import FULL_DECK
    return FULL_DECK


def abstract_to_incr(abstract: str, parsed: dict, my_pos: int,
                     ladder: tuple = NLHEState.RAISE_LADDER) -> str:
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

    fractions = dict(ladder)
    if abstract in fractions:
        raise_by = int(pot_after_call * fractions[abstract])
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
                 verbose: bool = False, harmonic: bool = False,
                 resolver=None, state_cls: type = NLHEState):
        self.agent = agent
        self.token = token
        self.verbose = verbose
        self.harmonic = harmonic
        self.resolver = resolver  # river_resolver.SubgameResolver or None
        self.state_cls = state_cls  # action profile the policy was trained on

    def _refresh_token(self, response: dict):
        if response.get("token"):
            self.token = response["token"]

    def _resolve_subgame(self, shadow, trace, response):
        """Re-solve the turn/river subgame; None on any failure -> blueprint."""
        from river_resolver import opponent_range
        try:
            our_hole = tuple(Card.new(c) for c in response["hole_cards"])
            rng = opponent_range(self.agent.policy, self.agent.bucketer,
                                 trace, 1 - shadow.to_act, our_hole,
                                 shadow.board_revealed())
            dist = self.resolver.resolve(shadow, our_hole, rng)
            actions = list(dist)
            choice = self.agent.rng.choices(
                actions, weights=[dist[a] for a in actions])[0]
            if self.verbose:
                pretty = {a: round(p, 3) for a, p in dist.items()}
                print(f"  street-{shadow.street} resolve: {pretty} -> {choice}")
            return choice
        except Exception as exc:  # live play must not die on a solver bug
            if self.verbose:
                print(f"  resolve failed ({exc}); using blueprint")
            return None

    def play_hand(self) -> int:
        """Plays one hand; the final server response is kept in self.last_hand
        (hole cards, board, action string, bot hole cards when shown down,
        winnings) so callers can log it for AIVAT-style luck adjustment."""
        r = _post("new_hand", {"token": self.token} if self.token else {})
        self._refresh_token(r)
        my_pos = r.get("client_pos")
        while "winnings" not in r or r.get("winnings") is None:
            action = r["action"]
            parsed = parse_action(action)
            if "error" in parsed:
                raise RuntimeError(f"parse error on {action!r}: {parsed['error']}")
            my_pos = r.get("client_pos", my_pos)
            trace = [] if self.resolver else None
            shadow = replay_abstract(action, r["hole_cards"], r["board"],
                                     harmonic=self.harmonic, trace_out=trace,
                                     state_cls=self.state_cls)
            if shadow is None or shadow.is_terminal():
                incr = "c" if parsed["last_bettor"] not in (-1, my_pos) else "k"
            else:
                abstract = None
                if (self.resolver is not None
                        and shadow.street >= self.resolver.from_street):
                    abstract = self._resolve_subgame(shadow, trace, r)
                if abstract is None:
                    abstract = self.agent.act(shadow)
                incr = abstract_to_incr(abstract, parsed, my_pos,
                                        ladder=self.state_cls.RAISE_LADDER)
            if self.verbose:
                print(f"  action={action!r} hole={r['hole_cards']} "
                      f"board={r['board']} -> {incr}")
            r = _post("act", {"token": self.token, "incr": incr})
            self._refresh_token(r)
        self.last_hand = {
            "winnings": r["winnings"],
            "client_pos": my_pos,
            "hole_cards": r.get("hole_cards"),
            "board": r.get("board"),
            "action": r.get("action"),
            "bot_hole_cards": r.get("bot_hole_cards"),
        }
        return r["winnings"]


def main():
    parser = argparse.ArgumentParser(description="Play the blueprint vs Slumbot")
    parser.add_argument("--hands", type=int, default=100)
    parser.add_argument("--blueprint", default="hunl_blueprint.pkl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-jsonl", default=None,
                        help="append one JSON record per hand (enables "
                             "aivat_report.py and crash-resume)")
    parser.add_argument("--translation", choices=("naive", "harmonic"),
                        default="naive",
                        help="opponent bet-size mapping: legacy thresholds "
                             "or pseudo-harmonic (Ganzfried & Sandholm)")
    parser.add_argument("--resolve", action="store_true",
                        help="re-solve late-street subgames instead of "
                             "playing the blueprint there")
    parser.add_argument("--resolve-iters", type=int, default=2000)
    parser.add_argument("--resolve-from", type=int, choices=(2, 3), default=3,
                        help="first street to re-solve: 2 = turn, 3 = river")
    args = parser.parse_args()

    keep_system_awake()

    with open(args.blueprint, "rb") as f:
        saved = pickle.load(f)
    bucketer = EquityBucketer(saved["buckets"], saved["samples"], args.seed,
                              mode=saved.get("mode", "ehs"))
    agent = PolicyAgent(saved["policy"], bucketer, seed=args.seed)
    resolver = None
    if args.resolve:
        from river_resolver import SubgameResolver
        resolver = SubgameResolver(args.resolve_iters, args.seed,
                                   from_street=args.resolve_from)
    state_cls = ACTION_PROFILES[saved.get("actions", "std")]
    session = SlumbotSession(agent, verbose=args.verbose,
                             harmonic=args.translation == "harmonic",
                             resolver=resolver, state_cls=state_cls)

    total, done = 0, 0
    if args.log_jsonl and os.path.exists(args.log_jsonl):
        with open(args.log_jsonl) as f:
            for line in f:
                total += json.loads(line)["winnings"]
                done += 1
        print(f"Resuming from {args.log_jsonl}: {done} hands, "
              f"{total:+} chips already played", flush=True)

    print(f"Playing hands {done + 1}..{args.hands} vs Slumbot "
          f"(blueprint: {saved['iterations']} MCCFR iterations)", flush=True)
    for h in range(done + 1, args.hands + 1):
        for attempt in range(5):
            try:
                winnings = session.play_hand()
                break
            except RuntimeError as exc:
                print(f"hand {h}: {exc}; retrying in {30 * (attempt + 1)}s",
                      flush=True)
                time.sleep(30 * (attempt + 1))
                session.token = None  # start a fresh server session
        else:
            raise RuntimeError(f"hand {h}: five consecutive failures, aborting")
        total += winnings
        if args.log_jsonl:
            with open(args.log_jsonl, "a") as f:
                f.write(json.dumps({"hand": h, **session.last_hand}) + "\n")
        if h % 100 == 0 or h == args.hands:
            mbb_per_hand = total / BIG_BLIND / h * 1000
            print(f"hand {h:>6}: total {total:>+9} chips, "
                  f"{mbb_per_hand:>+9.1f} mbb/hand", flush=True)
    print(f"\nFinal: {total:+} chips over {args.hands} hands = "
          f"{total / BIG_BLIND / args.hands * 1000:+.1f} mbb/hand "
          f"(Slumbot baseline: random ~ -50000, always-call ~ -3000, "
          f"strong bots > 0)", flush=True)


if __name__ == "__main__":
    main()
