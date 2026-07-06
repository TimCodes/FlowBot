# 6-Max No-Limit Hold'em Bot — Research Survey & Analysis

*Rung 5 of the FlowBot ladder. Survey compiled 2026-07-06 from the primary
literature, open-source implementations, and framework experiments run
against this repo's venv. Companion to the heads-up work in
`nlhe_engine.py` / `hunl_blueprint.py` / `slumbot_client.py`.*

---

## 1. Executive summary

There is exactly one publicly documented superhuman 6-max NLHE agent:
**Pluribus** (Brown & Sandholm, *Science* 2019). Its recipe is a modest
delta on what FlowBot already has for heads-up:

1. **Blueprint** via external-sampling MCCFR — the same algorithm as our
   rung-4 trainer — with two upgrades: **Linear CFR** discounting and
   **negative-regret pruning**. Trained in 8 days on a 64-core server
   (12,400 core-hours, < 512 GB RAM; famously ~$150 of cloud compute).
2. **Real-time depth-limited search** on top of the blueprint, where
   leaf nodes are evaluated by letting every player choose among **four
   continuation strategies** (the blueprint, plus fold-, call-, and
   raise-biased versions of it). Live play ran on 2 CPUs + 128 GB RAM.

The most important *conceptual* change from heads-up is theoretical:
with 3+ players, **CFR loses its Nash-equilibrium convergence
guarantee** entirely. What it retains is elimination of iteratively
strictly dominated strategies (Gibson 2013), and — empirically — the
strategies it produces are very strong anyway (Pluribus won 48 mbb/game
after AIVAT variance reduction over 10,000 hands vs. elite pros).
Practical consequence for us: **exploitability is no longer the metric;
money against a pool of opponents is.**

The most important *engineering* changes are: an N-player engine with
**side pots** (multiway all-ins), positional infoset keys (UTG ≠ button),
and a much larger game tree that makes pruning and abstraction quality
matter more.

Verified on this machine (see §5): RLCard's `no-limit-holdem` runs 6
players natively and is our best evaluation harness; OpenSpiel's
`universal_poker` is **not** in the Windows wheel, so our own engine
remains the training substrate — extend `nlhe_engine.py` to N players.

---

## 2. What actually changes from heads-up to 6-max

### 2.1 Theory: the equilibrium guarantee is gone

- In two-player zero-sum games, CFR's average strategy provably
  converges to an ε-Nash equilibrium. With ≥ 3 players this guarantee
  vanishes: regret minimization no longer implies Nash convergence
  ([Gibson 2013, arXiv:1305.0034](https://arxiv.org/abs/1305.0034)).
- Worse, even *playing* an exact Nash equilibrium in a 3+ player game
  carries no performance guarantee: if opponents deviate from the same
  equilibrium (or play a different one), your equilibrium share is not
  protected. Equilibrium selection is unresolved for multiplayer games.
- What survives: CFR provably eliminates iteratively strictly dominated
  actions/strategies in multiplayer games (Gibson), and empirically the
  Alberta group's CFR agents won 3-player limit events at the Annual
  Computer Poker Competition ([Abou Risk & Szafron, AAMAS 2010](https://poker.cs.ualberta.ca/publications/AAMAS10.pdf)).
- Pluribus' stance, worth adopting verbatim: stop arguing about Nash,
  produce a strategy by self-play, and let empirical results against
  strong opponents be the success criterion.

### 2.2 Practice: game size, position, and multiway dynamics

- **Tree size**: each betting round can now contain up to 6 acting
  players and re-raising chains between any pair; infoset counts grow
  by orders of magnitude over HU at the same abstraction fidelity.
  External sampling + pruning is what keeps traversal affordable.
- **Position matters structurally**: the infoset key must encode seat /
  players-still-in / position relative to the button, not just "P0/P1".
  A preflop AKo open is a different decision UTG (4 players behind)
  vs. button.
- **Side pots**: with 3+ players, one all-in no longer ends the hand.
  The engine needs per-player contribution tracking and layered side-pot
  settlement — this is the single biggest engine change (code in §6.4).
- **Multiway equities compress**: hand values shift (suited/connected
  playable hands gain, dominated offsuit broadways lose), so preflop
  bucketing tuned on HU equities mis-ranks hands. Equity rollouts for
  abstraction must be computed **against 5 random hands**, not 1.
- **No public benchmark bot**: Slumbot is heads-up only. 6-max
  evaluation means self-built opponent pools, duplicate dealing, and
  AIVAT-style variance reduction (§7).

---

## 3. The Pluribus recipe in detail

Primary sources: [Science paper](https://www.science.org/doi/10.1126/science.aay2400)
([author PDF](https://noambrown.com/papers/19-Science-Superhuman.pdf)),
[supplementary material](https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf),
[Depth-Limited Solving paper (arXiv:1805.08195)](https://arxiv.org/abs/1805.08195).

### 3.1 Blueprint: Linear ES-MCCFR with negative-regret pruning

Same skeleton as our `hulhe_mccfr.py` trainer, with three deltas:

1. **Linear CFR**: on iteration *t*, new regret and average-strategy
   contributions are weighted by *t* (equivalently, accumulated sums are
   discounted by *t/(t+1)* each iteration). Early garbage washes out
   ~quadratically faster. Pluribus applied the discounting only for the
   first 400 minutes of the run — after that the multiply wasn't worth
   its cost.
2. **Negative-regret pruning**: an action whose cumulative regret is
   below a large negative threshold (−300,000,000 in Pluribus) is
   skipped in **95% of iterations**; the other 5% of iterations traverse
   everything so wrongly-pruned actions can recover. This is both a
   large speedup (most of the tree is bad actions) and overflow
   protection. Regrets are floored so recovery stays possible.
3. **Multi-valued terminals**: payoffs are a *vector* (one entry per
   player), not a scalar — there is no "opponent utility = −u" shortcut
   in a 6-player game. The traverser updates regrets from its own
   component. (Chip-EV is still constant-sum across the 6 seats, but
   pairwise it is general-sum.)

Scale of the real thing, for calibration: 8 days × 64 cores, < 512 GB
RAM, blueprint played only as the *preflop* strategy (and as the search
prior thereafter). Pluribus' blueprint action abstraction allowed up to
14 bet sizes at some nodes; its information abstraction used k-means
bucketing on equity-distribution features (KE/KO-style, as in Libratus),
with imperfect recall between streets — the same family as our
`card_abstraction.py`, just bigger and computed vs. 5 opponents.

### 3.2 Real-time depth-limited search

The core problem: subgame solving with a scalar value function at the
depth limit is unsound in imperfect-information games (the opponent's
best response below the limit depends on *your* strategy above it).
Brown's fix ([arXiv:1805.08195](https://arxiv.org/abs/1805.08195)): at
each leaf of the lookahead tree, let **each player pick among k
continuation strategies** for the remainder of the game, and solve the
enlarged game. Pluribus used **k = 4**: the blueprint, and three
modified blueprints biased toward folding, calling, or raising —
implemented by multiplying the biased action's probability by 5 and
renormalizing (§6.3). This forces the search strategy to be robust to
a *range* of below-limit opponent behaviors, not a point estimate.

Operational details worth copying:

- Pluribus plays the **blueprint directly on the first betting round**
  (preflop abstraction is fine-grained enough) and **searches on every
  later decision**, and also whenever an opponent takes an off-tree bet
  size — search *is* the action-translation mechanism, replacing the
  pseudo-harmonic mapping we currently use in `slumbot_client.py`.
- To remain unpredictable, Pluribus keeps its **own** hand's strategy
  balanced across its entire range at the start of the subgame (it
  solves for all hands it *could* hold, not just the one it has).
- Search used ~6 bet sizes per node (vs. up to 14 in the blueprint),
  and Monte Carlo rollouts weighted by the blueprint to seed leaf values.
- Live-play footprint: 2 CPUs, 128 GB RAM, ~20 s per hand on average —
  the search, not the blueprint, is what made Pluribus superhuman-cheap.

### 3.3 Evaluation

10,000 hands vs. professionals over 12 days (5 pros + Pluribus per
table, and the inverse 1-pro-vs-5-copies format). Result after AIVAT:
**+48 mbb/game, standard error 25 mbb/game** — statistically significant
at the level the study aimed for, only because AIVAT cut variance
enormously. Lesson for us: build variance reduction into the eval
harness from day one (§7).

---

## 4. Beyond Pluribus: the rest of the field

| Approach | Representative work | Relevance to a 6-max bot |
|---|---|---|
| Tabular CFR + abstraction + search | Libratus, **Pluribus** | The only proven 6-max recipe; our path. |
| Continual re-solving w/ NN values | DeepStack (2017), Student of Games | Sound for 2p zero-sum; value nets for 6 players are an open problem (input = 6 ranges, equilibrium selection unclear). |
| RL + search on public belief states | ReBeL (2020) | Same 2p-zero-sum soundness caveat; elegant but doesn't transfer directly. |
| End-to-end deep RL, no search | [AlphaHoldem (AAAI 2022)](https://ojs.aaai.org/index.php/AAAI/article/view/20394) — pseudo-siamese net, "trinal-clip" PPO, beats Slumbot HU after ~3 days on 1 PC | Attractive compute profile; multiplayer version (so-called AlphaHoldem-6 experiments) far less validated. Good *opponent-pool member*, not the backbone. |
| Deep CFR family | Deep CFR, DREAM, [Deep Predictive-Discounted CFR (2025)](https://arxiv.org/abs/2511.08174) | Replaces tabular regrets with nets — useful if our abstraction becomes the bottleneck; adds training instability. |
| LLM-based agents | PokerGPT (2024), SpinGPT (2025), [PokerBench / "how far are LLMs" (2026)](https://www.arxiv.org/pdf/2602.00528) | Consistently below CFR-based solvers; interesting for opponent modeling / exploitation layers, not for the equilibrium core. |
| Exploitation-oriented agents | [Beyond GTO: profit-maximizing agents (2025)](https://arxiv.org/pdf/2509.23747), restricted Nash response lineage | The publishable-delta direction for rung 5b: 6-max pools are full of weak players; equilibrium-ish base + opponent adaptation is where money (and novelty) is. |

Analysis: for a 6-player game the CFR-blueprint-plus-search recipe is
still the only approach with a superhuman receipt attached. Deep RL and
LLM agents are worth having in the *evaluation pool* precisely because
they play differently, which stress-tests robustness in ways self-play
snapshots don't.

---

## 5. Open-source landscape (verified where possible)

| Repo / framework | Language, stars | What it offers | Verdict for FlowBot |
|---|---|---|---|
| [fedden/poker_ai](https://github.com/fedden/poker_ai) | Python, ★1.6k | Fullest open Pluribus attempt: clustering CLI, MCCFR blueprint, resume, play CLI. **Limitation: shipped tuned for a 20-card deck**; full-deck clustering is expensive and less tested. | Read for architecture (lut/clustering module boundaries); don't depend on it. |
| [ozzi7/Poker-MCCFRM](https://github.com/ozzi7/Poker-MCCFRM) | C# + C++, ★112 | N-player MCCFR following the Pluribus paper: Waugh's hand-isomorphism indexer, EMD k-means flop/turn buckets, **OCHS river buckets**, live search in the C++ version. Wants >128 GB RAM at full size. | Best reference for the *abstraction pipeline* (OCHS + EMD details, imperfect recall postflop). |
| [whatsdis/pluribus](https://github.com/whatsdis/pluribus) | Python, ★238 | Direct implementation from the Science supplementary pseudocode (linear CFR, pruning, search skeleton). | Best reference for the *algorithm*, cleanest mapping paper→code. |
| [RLCard](https://github.com/datamllab/rlcard) | Python | `no-limit-holdem` env with configurable player count. **Verified working in this repo's venv** (see below). Coarse action abstraction (5 actions). | Our 6-max *evaluation harness* and baseline-agent pool; too abstracted to be the training engine. |
| [OpenSpiel universal_poker](https://github.com/deepmind/open_spiel) | C++/Python | ACPC-based parameterized N-player NLHE. **Verified absent from the Windows cp313 wheel** — `pyspiel.load_game('universal_poker', ...)` raises `Unknown game` in this venv (only `kuhn_poker`/`leduc_poker` ship). | Unavailable here without WSL/self-build. Reinforces: extend our own `nlhe_engine.py`. |
| [PokerBotAI/awesome-poker-ai](https://github.com/PokerBotAI/awesome-poker-ai) | — | Curated index of papers/tools/bots. | Bookmark. |

Verified RLCard 6-player session from this repo's venv (2026-07-06):

```python
import rlcard

env = rlcard.make('no-limit-holdem',
                  config={'game_num_players': 6, 'seed': 0})
print(env.num_players)      # -> 6
print(env.num_actions)      # -> 5  (fold, check/call, half-pot, pot, all-in)
state, first_player = env.reset()
# first to act preflop is seat 3 (UTG with blinds in 0/1... RLCard uses
# dealer-relative seating); state['legal_actions'] lists the legal subset.
```

Output observed: `6 players, action space 5`, first to act seat 3,
legal actions `{0: fold, 1: call, 3: pot, 4: all-in}` — note RLCard
already prunes illegal raise sizes, matching our engine's convention.

---

## 6. Code samples

The samples below are written against this repo's conventions (same
state API as `nlhe_engine.py`, treys for showdowns) so they can be
lifted into rung-5 modules with minimal editing.

### 6.1 Multiplayer external-sampling MCCFR with Linear CFR + pruning

The rung-4 trainer's traversal, upgraded for N players. Three changes:
vector payoffs, seat-aware infoset keys, and the two Pluribus tricks.

```python
import random
import numpy as np

PRUNE_THRESHOLD = -3.0e8      # Pluribus: -300M cumulative regret
PRUNE_PROB      = 0.95        # skip pruned actions in 95% of iterations
REGRET_FLOOR    = -3.1e8      # keep recovery possible, prevent overflow

class MultiwayMCCFR:
    def __init__(self, engine_factory, num_players=6):
        self.engine_factory = engine_factory
        self.num_players = num_players
        self.regret = {}        # infoset -> np.array over actions
        self.strat_sum = {}     # infoset -> np.array over actions

    def _sigma(self, infoset, n_actions):
        r = np.maximum(self.regret.setdefault(
            infoset, np.zeros(n_actions)), 0.0)
        total = r.sum()
        return r / total if total > 0 else np.full(n_actions, 1.0 / n_actions)

    def train_iteration(self, t):
        """One Linear-CFR iteration: every seat takes a turn as traverser."""
        prune = random.random() < PRUNE_PROB
        for seat in range(self.num_players):
            state = self.engine_factory()          # fresh shuffled hand
            self._traverse(state, seat, t, prune)

    def _traverse(self, state, traverser, t, prune):
        if state.is_terminal():
            return np.asarray(state.payoffs())     # length-6 vector, chips
        p = state.current_player()
        actions = state.legal_actions()
        # Key change vs heads-up: the infoset encodes *position and
        # who is still in*, e.g. (bucket, street, btn_offset, alive_mask,
        # betting_history) — not just the two-seat history string.
        infoset = state.infoset_key(p)
        sigma = self._sigma(infoset, len(actions))

        if p != traverser:
            # external sampling: sample one action for every other seat
            a = random.choices(range(len(actions)), weights=sigma)[0]
            return self._traverse(state.child(actions[a]), traverser, t, prune)

        node_util = np.zeros(self.num_players)
        action_util = np.zeros(len(actions))
        explored = []
        for i, a in enumerate(actions):
            if prune and self.regret[infoset][i] < PRUNE_THRESHOLD:
                continue                            # negative-regret pruning
            child_util = self._traverse(state.child(a), traverser, t, prune)
            action_util[i] = child_util[traverser]  # own component only
            node_util += sigma[i] * child_util
            explored.append(i)
        for i in explored:
            # Linear CFR: weight this iteration's contribution by t
            self.regret[infoset][i] = max(
                REGRET_FLOOR,
                self.regret[infoset][i]
                + t * (action_util[i] - node_util[traverser]))
        ss = self.strat_sum.setdefault(infoset, np.zeros(len(actions)))
        ss += t * sigma
        return node_util
```

Analysis notes:
- Weighting by *t* directly (instead of discounting the stores) is the
  numerically simplest Linear CFR form; switch to periodic multiplicative
  discounting if the sums grow too fast, and stop discounting after a
  warmup phase as Pluribus did.
- With vector payoffs the traversal cost per iteration is ~3× HU at
  equal depth just from extra live seats; pruning typically wins that
  back and more once regrets separate.
- The average strategy (`strat_sum`) is what you play; per-street
  purification (play argmax when one action dominates) reduced
  exploitability noise in Pluribus' preflop play.

### 6.2 Preflop abstraction must be recomputed for 6-max

The 169 preflop classes stay lossless, but bucket *features* change:
equity vs. 5 random hands, not 1. Small delta to `card_abstraction.py`:

```python
from treys import Card, Deck, Evaluator

_ev = Evaluator()

def multiway_equity(hole, n_opponents=5, n_samples=2000, rng=None):
    """MC equity of `hole` vs n independent random hands to showdown.
    HU-tuned buckets mis-rank hands multiway: e.g. KJo drops far more
    than 87s when three extra players see the flop."""
    rng = rng or random.Random(0)
    wins = 0.0
    for _ in range(n_samples):
        deck = Deck(); deck.shuffle(rng)          # exclude `hole` cards
        deck.cards = [c for c in deck.cards if c not in hole]
        opps = [deck.draw(2) for _ in range(n_opponents)]
        board = deck.draw(5)
        mine = _ev.evaluate(board, hole)
        theirs = min(_ev.evaluate(board, o) for o in opps)   # best opp
        if mine < theirs:   wins += 1.0            # treys: lower = better
        elif mine == theirs: wins += 0.5
    return wins / n_samples
```

For postflop buckets the proven pipeline (ozzi7, Libratus, Pluribus) is:
equity-distribution histograms clustered with earth-mover's-distance
k-means on flop/turn, **OCHS** (opponent-cluster hand strength) on the
river, imperfect recall between streets — exactly our current
architecture with the rollout opponent count changed and k raised.

### 6.3 Depth-limited search: the four continuation strategies

The heart of Pluribus' search, and small enough to state exactly. Given
a blueprint policy at an infoset, the biased continuations are:

```python
def biased_policy(blueprint_probs, actions, bias, factor=5.0):
    """Pluribus leaf continuation: multiply the biased action class's
    probability by `factor`, renormalize. bias in
    {'fold','call','raise',None} — None returns the blueprint itself."""
    w = np.array(blueprint_probs, dtype=float)
    for i, a in enumerate(actions):
        if bias == 'fold' and a.kind == 'fold':      w[i] *= factor
        elif bias == 'call' and a.kind == 'call':    w[i] *= factor
        elif bias == 'raise' and a.is_raise:         w[i] *= factor
    return w / w.sum()

CONTINUATIONS = [None, 'fold', 'call', 'raise']      # k = 4
```

At each leaf of the lookahead tree, the subgame solver adds one chance-
free decision node per player choosing among these 4 continuations, and
leaf values are blueprint-weighted rollouts under the chosen tuple. The
solved subgame strategy is then robust against opponents who play
*meaningfully differently* below the depth limit — this is what makes
depth-limited solving sound-ish where a scalar value function is not
([arXiv:1805.08195](https://arxiv.org/abs/1805.08195)).

### 6.4 The N-player engine delta: side pots

The one piece of pure engineering with no HU analogue. Settlement with
per-player contributions (drop-in for a rung-5 `nlhe6_engine.py`):

```python
def settle(contribs, folded, hand_ranks):
    """contribs: chips each seat put in (list[int]); folded: set of seats;
    hand_ranks: seat -> showdown rank (lower better), only for non-folded.
    Returns net payoff vector. Handles arbitrary layered side pots."""
    n = len(contribs)
    payoff = [-c for c in contribs]                 # everyone pays in
    remaining = list(contribs)
    while any(r > 0 for r in remaining):
        layer = min(r for r in remaining if r > 0)  # next all-in level
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
        for j, s in enumerate(sorted(winners)):     # odd chips to earliest
            payoff[s] += share + (1 if j < odd else 0)
    return payoff
```

Everything else in `nlhe_engine.py` generalizes mechanically: stacks,
street-cumulative bets, and the fold/call/½-pot/pot/all-in abstraction
become per-seat arrays; the betting round ends when action returns to
the last aggressor with all live non-all-in seats matched.

### 6.5 Evaluation harness sketch: duplicate 6-max in RLCard

```python
import itertools, rlcard

def duplicate_match(agents, n_deals=500, seed=0):
    """Rotate each agent through all 6 seats on the same shuffled deals
    (duplicate poker): kills most positional/card variance for free,
    the poor man's AIVAT."""
    env = rlcard.make('no-limit-holdem',
                      config={'game_num_players': 6, 'seed': seed})
    totals = {id(a): 0.0 for a in agents}
    for deal in range(n_deals):
        for rot in range(6):                         # 6 seatings per deal
            seating = agents[rot:] + agents[:rot]
            env.seed(seed + deal)                    # same cards each rot
            env.set_agents(seating)
            _, payoffs = env.run(is_training=False)
            for seat, a in enumerate(seating):
                totals[id(a)] += payoffs[seat]
    return totals   # convert to mbb/hand with the env's big blind
```

---

## 7. Evaluating a 6-max bot (no Slumbot exists here)

1. **Opponent pool, not a single benchmark**: random / call-station /
  earlier blueprint snapshots / RLCard's DQN & NFSP baselines /
  (later) an AlphaHoldem-style PPO agent. Track mbb/hand vs. each pool
  member *and* in mixed tables — 6-max strength is not transitive.
2. **Duplicate dealing** (§6.5) as the default variance reducer.
3. **AIVAT** ([Burch et al.](https://arxiv.org/abs/1612.06915)) once
  real measurements matter: with a value estimate at every decision it
  cut Pluribus' eval variance enough to make 10k hands significant —
  ±25 mbb/game instead of hundreds.
4. Report **mbb/game with standard errors**, seat-balanced, ≥100k
  simulated hands for self-pool numbers (cheap) even if human/live
  numbers stay small.

---

## 8. Recommended FlowBot rung-5 roadmap

| Step | Deliverable | Success criterion |
|---|---|---|
| 5.1 | `nlhe6_engine.py`: N-player engine + side pots (§6.4), positional infoset keys; tests incl. multiway all-in settlement edge cases | Test suite green; hand histories audited by eye |
| 5.2 | Multiway abstraction: 169 preflop + EMD buckets from 5-opponent rollouts (§6.2) | AA≫72o sanity multiway; monotone bucket equities |
| 5.3 | 6-max blueprint: Linear ES-MCCFR + pruning (§6.1) | Beats random & call pools; positive vs rung-4 policy transplanted to 6-max |
| 5.4 | Duplicate-poker eval harness in RLCard + snapshot pool (§6.5) | mbb/hand curves with SEs across ≥3 opponent types |
| 5.5 | Depth-limited search with 4 continuations (§6.3) | Search agent beats its own blueprint by a clear margin |
| 5.6 | Novelty: opponent-exploitation layer on the blueprint (Beyond-GTO direction, §4) | Publishable delta: exploits weak pool members without collapsing vs strong ones |

Compute reality check: Pluribus-scale is 12,400 core-hours, but the
*shape* of the curve is friendly — our 6-minute HU blueprint already
beat trivial agents by thousands of mbb, and a 6-max blueprint in the
tens of millions of ES-MCCFR touches (days on this workstation, with
pruning) should comfortably clear step 5.3's bar. The search layer, not
blueprint scale, is where superhuman play came from.

---

## 9. Sources

**Papers**
- Brown & Sandholm, [*Superhuman AI for multiplayer poker*](https://www.science.org/doi/10.1126/science.aay2400), Science 2019 — [author PDF](https://noambrown.com/papers/19-Science-Superhuman.pdf), [supplementary](https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf)
- Brown, Sandholm & Amos, [*Depth-Limited Solving for Imperfect-Information Games*](https://arxiv.org/abs/1805.08195), NeurIPS 2018
- Gibson, [*Regret Minimization in Non-Zero-Sum Games…*](https://arxiv.org/abs/1305.0034), 2013 (multiplayer CFR theory)
- Abou Risk & Szafron, [*Using CFR to Create Competitive Multiplayer Poker Agents*](https://poker.cs.ualberta.ca/publications/AAMAS10.pdf), AAMAS 2010
- Zhao et al., [*AlphaHoldem*](https://ojs.aaai.org/index.php/AAAI/article/view/20394), AAAI 2022
- Burch et al., [*AIVAT*](https://arxiv.org/abs/1612.06915), 2017
- [*Beyond Game Theory Optimal: Profit-Maximizing Poker Agents*](https://arxiv.org/pdf/2509.23747), 2025; [*How far are LLMs from professional poker players?*](https://www.arxiv.org/pdf/2602.00528), 2026; [*Deep Predictive-Discounted CFR*](https://arxiv.org/abs/2511.08174), 2025
- Ganzfried & Sandholm on action translation (pseudo-harmonic mapping) — already cited in the rung-4 notes

**Code**
- [fedden/poker_ai](https://github.com/fedden/poker_ai) · [ozzi7/Poker-MCCFRM](https://github.com/ozzi7/Poker-MCCFRM) · [whatsdis/pluribus](https://github.com/whatsdis/pluribus)
- [RLCard](https://github.com/datamllab/rlcard) · [OpenSpiel](https://github.com/deepmind/open_spiel) · [awesome-poker-ai](https://github.com/PokerBotAI/awesome-poker-ai)

**Talks / video**
- Noam Brown, [*Parables on the Power of Planning in AI: From Poker to Diplomacy*](https://www.youtube.com/watch?v=eaAonE58sLU) (UW Allen School Distinguished Lecture, 2024) — the best single hour on why search beat scale in poker
- Noam Brown podcast interviews: [TalkRL/gradient-style S3E14](https://www.youtube.com/watch?v=ceCg90Q9N6Y), [SuperDataScience 569](https://www.youtube.com/watch?v=5syG8Zkekx8)

**Articles**
- [Scientific American: AI Conquers Six-Player Poker](https://www.scientificamerican.com/article/ai-conquers-six-player-poker/) · [CMU press release](https://www.cmu.edu/news/stories/archives/2019/july/cmu-facebook-ai-beats-poker-pros.html) · [LessWrong close-read of the paper](https://www.lesswrong.com/posts/6qtq6KDvj86DXqfp6/let-s-read-superhuman-ai-for-multiplayer-poker) · [KDnuggets retrospective](https://www.kdnuggets.com/2020/12/remembering-pluribus-facebook-master-difficult-poker-game.html) · [Pluribus explained for builders](https://openpoker.ai/blog/pluribus-poker-bot-explained)

**Scope note:** research/benchmark use only — deploying bots on
real-money sites violates their ToS and may be illegal in some
jurisdictions.
