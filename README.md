# FlowBot — poker bot research

Working through the research ladder for a Texas Hold'em bot, following the
CFR lineage (Cepheus → DeepStack → Libratus → Pluribus). Each rung has a
verifiable success criterion before moving to the next.

## Ladder

| Rung | Game | Method | Status | Success criterion |
|---|---|---|---|---|
| 1 | Kuhn poker | Vanilla CFR + CFR+ (pure stdlib) | ✅ done | Match analytical equilibrium; NashConv → 0 |
| 2 | Leduc Hold'em | Tabular CFR / CFR+ / MCCFR (RLCard + OpenSpiel) | ✅ done | Exact exploitability curves via OpenSpiel best response |
| 3 | Heads-up **limit** Hold'em | External-sampling MCCFR + equity bucketing | ✅ done | Beats baseline agents (mbb/hand); exact exploitability no longer tractable |
| 4 | Heads-up **no-limit** Hold'em | MCCFR blueprint (f/c/½pot/pot/all-in) + live Slumbot client | ✅ done | **−5.5 mbb/hand vs Slumbot** (1000 hands, 2.5M-iter blueprint); next: 10k+ hands + AIVAT, subgame re-solving |
| 5 | **6-max** no-limit Hold'em | Pluribus recipe: N-player Linear-MCCFR blueprint + depth-limited search; opponent exploitation | 🟡 blueprint done | Beats agent pool (seat-rotated mbb/hand); next: big run, depth-limited search, AIVAT — survey & plan in [research_6max.md](research_6max.md) |

## Files

- `kuhn_cfr.py` — chance-sampling CFR and CFR+ for Kuhn poker, plus an exact
  best-response oracle (`nash_conv`) for measuring exploitability.
  Run: `python kuhn_cfr.py --iterations 200000 [--plus]`
- `test_kuhn_cfr.py` — 19 tests verifying convergence to the known
  one-parameter equilibrium family (α relations, 1/3 call/bluff frequencies,
  game value −1/18, NashConv < 0.02). Run: `python -m unittest test_kuhn_cfr -v`
- `leduc_cfr_rlcard.py` — tabular CFR on Leduc Hold'em via RLCard, evaluated
  by tournament vs a random agent (smoke test only).
  Run: `.venv\Scripts\python leduc_cfr_rlcard.py --episodes 2000`
- `leduc_exploitability.py` — the principled rung-2 metric: exact
  exploitability convergence curves for vanilla CFR, CFR+, and
  external-sampling MCCFR on OpenSpiel's `leduc_poker`; writes
  `leduc_exploitability.csv`.
  Run: `.venv\Scripts\python leduc_exploitability.py`
  Note: RLCard's and OpenSpiel's Leduc implementations differ slightly in
  encoding, so the exploitability run trains fresh solvers in OpenSpiel
  rather than importing the RLCard policy.

## Environment

- Rung 1 is pure stdlib (any Python ≥ 3.10).
- Rung 2 uses the venv: `python -m venv .venv`, then
  `.venv\Scripts\pip install rlcard setuptools open_spiel`.
  (OpenSpiel ≥1.6 ships native Windows wheels, including cp313.)
  - `setuptools` is required on Python 3.12+ because rlcard 1.2.0 imports the
    removed `distutils`.
  - rlcard 1.2.0 also calls `ndarray.tostring()` (removed in NumPy 2.0);
    `leduc_cfr_rlcard.py` ships a `CompatCFRAgent` shim, so the installed
    package is untouched.

## Reference results (seed 42, 200k iterations, vanilla CFR)

```
Sampled game value for P0: -0.0544   (analytical: -0.0556)
NashConv (exploitability): 0.00185
P0 opens J 0.277 (=α), Q 0.000, K 0.837 (≈3α)
P0 calls check-raise with Q 0.619 (≈ α + 1/3)
P1 vs bet: folds J, calls Q 0.335 (≈1/3), calls K 1.000
```

## Rung 3 files

- `holdem_engine.py` — heads-up limit Hold'em rules engine (blinds 1/2,
  small/big bets 2/4, 4-bet cap per street), treys showdown evaluation.
- `card_abstraction.py` — 169 lossless preflop classes + Monte Carlo
  expected-hand-strength buckets postflop (imperfect-recall abstraction;
  docstring notes the E[HS²]/potential-aware upgrades for later).
- `hulhe_mccfr.py` — external-sampling MCCFR trainer over the abstracted
  game, PolicyAgent/RandomAgent/CallAgent, seat-alternated match runner
  reporting mbb/hand.
  Run: `.venv\Scripts\python hulhe_mccfr.py --iterations 30000`
- `test_hulhe.py` — 19 tests: betting rules (BB option, bet cap, street
  bet sizes), showdown/fold payoffs, abstraction sanity (AA ≫ 72o, royal
  flush hits the top bucket), trainer smoke + beats-random + fair-match-runner.
  Run: `.venv\Scripts\python -m unittest test_hulhe -v`

## Rung 4 files

- `nlhe_engine.py` — heads-up no-limit engine, ACPC/Slumbot conventions
  (blinds 50/100, 200 BB stacks). The action abstraction lives in the engine:
  fold / call / half-pot / pot / all-in, with min-raise pruning and
  near-stack raises collapsing into all-in. Same state API as the limit
  engine, so the MCCFR trainer is shared.
- `hunl_blueprint.py` — trains the Libratus-style blueprint with the shared
  ES-MCCFR trainer. Run: `.venv\Scripts\python hunl_blueprint.py`
- `slumbot_client.py` — HTTP bridge to slumbot.com: ports the official
  action-string parser (with per-position contribution tracking), maps real
  bets to abstract sizes by pot ratio (naive thresholds; pseudo-harmonic
  translation is the documented upgrade), replays a shadow abstract state
  per decision, and converts the policy's abstract action back to a legal
  `bX` increment. Run: `.venv\Scripts\python slumbot_client.py --hands 100`
- `test_nlhe.py` (14 tests) + `test_slumbot_client.py` (23 tests, offline).

## Rung 5 files

- `research_6max.md` — survey of the Pluribus recipe, multiplayer CFR
  theory, open-source landscape, and the 5.1–5.6 roadmap this rung follows.
- `nlhe6_engine.py` — 3–6 player no-limit engine (blinds 50/100, 200 BB
  stacks, same f/c/½pot/pot/all-in abstraction). Multiway deltas: a
  `pending` counter drives betting-round closure (BB option, re-opened
  action), and `settle()` distributes layered side pots, dead money, and
  odd chips. Same state API as the heads-up engines plus vector `payoffs()`.
- `nlhe6_mccfr.py` — Linear external-sampling MCCFR (Pluribus blueprint
  variant): per-seat traversals with vector payoffs, iteration-weighted
  (linear) regret/strategy updates, negative-regret pruning with recovery
  iterations, seat-prefixed infoset keys (seat = position). Includes
  `PolicyAgent6` and `play_table` (hero rotates through all six seats).
- `nlhe6_blueprint.py` — trains the 6-max blueprint with resumable atomic
  checkpoints; equity buckets roll out vs 5 random opponents (`--opponents`),
  optional E[HS²] bucketing (`--mode ehs2`).
  Run: `.venv\Scripts\python nlhe6_blueprint.py --iterations 30000`
- `test_nlhe6.py` — 25 tests: side-pot settlement (layered all-ins, dead
  money, odd-chip splits), multiway betting flow (BB option, squeeze
  re-opens action, fold-around, fast-forward runouts), multiway equity
  sanity (AA ≫ 72o five-way; five-way < heads-up equity), trainer smoke
  (beats a random table, pruning floor respected), chip conservation.
  Run: `.venv\Scripts\python -m unittest test_nlhe6 -v`
- `resume_training6.cmd` — detached relaunch for the big blueprint run
  (2M iters, 12 buckets, resumable checkpoints every 50k).
- `nlhe6_search.py` — rung 5.5: Pluribus-style real-time depth-limited
  search. Blueprint plays preflop; postflop the agent re-solves the
  subgame rooted at the current public state with ES-MCCFR: opponents'
  hole ranges are reach-weighted by replaying their observed actions
  through the blueprint, each iteration re-deals unrevealed cards
  (determinization), the solve is depth-limited to the end of the current
  street, and at each leaf every live player picks among four continuation
  strategies — the blueprint, or fold-/call-/raise-biased versions of it
  (biased action ×5, renormalized) — as a regret-minimized meta-decision.
  Hero's own hand is fixed ("unsafe" search); range-balanced solving is
  the documented upgrade.
  Run: `.venv\Scripts\python nlhe6_search.py --blueprint nlhe6_blueprint_big.pkl --hands 200`
- `test_nlhe6_search.py` — 13 tests: continuation biasing math, history
  replay, reach-weighted ranges (raisers weighted onto strong buckets),
  river solves need no leaf nodes, flop solves create them, determinization
  card-safety, agent legality end-to-end.
  Run: `.venv\Scripts\python -m unittest test_nlhe6_search -v`

Evaluation note: there is no 6-max Slumbot, so the metric is mbb/hand vs
agent pools with the hero rotated through every seat. With 3+ players CFR
keeps no Nash guarantee (only dominated-strategy elimination), so pool
results *are* the success criterion, not a proxy for exploitability.

## Reference results — big 6-max blueprint (2M iterations, 12 buckets vs-5-opponent EHS, prune after 20k, seed 0)

```
iter  100,000:  8,674,615 infosets   vs 5 random  +7507 mbb/hand   vs 5 call  +2097 mbb/hand
iter  500,000: 24,148,000 infosets   (curves oscillate in the bands below)
iter 2,000,000: 53,153,572 infosets  vs 5 random band +2400..+17000, vs 5 call band +1600..+13100
```

11.6 h wall time (~48 iters/s sustained), 2.5 GB policy (`nlhe6_blueprint_big.pkl`),
5.4 GB trainer state. Unlike heads-up (saturated at 1.4M infosets), the 6-max
tree was still adding ~15k infosets per 1k iterations at 2M — abstraction
capacity, not training time, is the binding constraint now. Checkpoint evals
are 4000 hands and swing by thousands of mbb (6-way 200 BB pots); both pool
metrics stayed positive from 50k onward.

## Reference results — 6-max blueprint (10k iterations, 8 buckets vs-5-opponent EHS, prune after 4k, seed 0)

```
iter  2500:   441,858 infosets   vs 5 random   +369 mbb/hand   vs 5 call +19930 mbb/hand
iter  5000:   854,749 infosets   vs 5 random  +5408 mbb/hand   vs 5 call  -2660 mbb/hand
iter 10000: 1,531,929 infosets   vs 5 random +10440 mbb/hand   vs 5 call   +594 mbb/hand
```

~5 min wall time; blueprint in `nlhe6_blueprint.pkl`. The 6-max game builds
infosets ~5x faster than heads-up at equal iterations (six seats × more
histories) and is far from saturation at 10k. Eval noise is much larger than
heads-up: 6-way 200 BB pots mean a single all-in swing is ±200,000 mbb, so
2000-hand evals carry several-thousand-mbb standard error — the vs-call
wobble at 5k is mostly that, plus a genuinely raise-happy transitional
policy (against 5 call-stations every bluff is called down). Both pools are
comfortably beaten by 10k. A serious blueprint needs millions of iterations
(see the HUNL 2.5M run) plus the rung-5.5 search layer.

## Reference results — HUNL blueprint (30k iterations, 8 buckets, seed 0)

```
iter 10000: 503,048 infosets   vs random +3580 mbb/hand   vs call  +9390 mbb/hand
iter 30000: 707,668 infosets   vs random +5950 mbb/hand   vs call +16797 mbb/hand
```

~6 min wall time; blueprint in `hunl_blueprint.pkl`. No-limit mbb numbers are
an order of magnitude above the limit game's because pots are 200 BB deep.

**First live Slumbot result (100-hand smoke, 2026-07-03):** −4795 mbb/hand.
Expected for a 6-minute blueprint: clearly losing to a near-equilibrium bot,
but far above the ~−50,000 of a random agent (and ~−750 of always-fold).
100 hands carries ~±1000 mbb/hand noise — treat this as a pipeline test, not
a measurement.

## Reference results — big blueprint (2.5M iterations, 12 buckets)

Trained in ~10 h of compute (survived one OS restart via `--resume`; see
`resume_training.cmd`). 1,455,924 infosets, checkpoint `hunl_blueprint_big.pkl`.
Vs-baseline margins *declined* over training (+10.6k → +5.3k vs random) —
the average policy converging toward balanced play, not a regression.

**Slumbot, 1000 hands (2026-07-05): −5.5 mbb/hand** — statistically
indistinguishable from break-even (±~1500 mbb at this sample size), versus
−4795 for the 30k-iteration blueprint.

**Slumbot, 10,000 hands (2026-07-07), blueprint only:**

```
                              raw:  −332.4 ± 185.7 mbb/hand   (per-hand sd 18,571)
                  all-in adjusted:  −250.7 ± 157.8 mbb/hand
AIVAT-lite (all-in + preflop OLS):  −229.8 ± 157.6 mbb/hand
```

The big sample corrects the 1000-hand illusion: the blueprint alone is
clearly losing (~−230 luck-adjusted, and it ran ~100 mbb/hand unlucky in
all-ins). AIVAT-lite (`aivat_report.py`) gives an unbiased 1.4× variance
cut; the ~10× of full AIVAT needs learned value functions at decision nodes.

## Rung 4 upgrades (commit 1387119)

- **E[HS²] bucketing** (`--mode ehs2`): RMS E[HS²] via a two-opponent
  product estimator; polarized draws now separate from static made hands.
  Retrained 2.5M-iteration blueprint: `hunl_blueprint_hs2.pkl`.
- **Pseudo-harmonic action translation** (`--translation harmonic`):
  Ganzfried & Sandholm randomized bet mapping, deterministic per bet.
- **River subgame re-solving** (`--resolve`): blueprint-reach-weighted
  opponent range + exact ES-CFR solve of each river subgame
  (`river_resolver.py`); falls back to the blueprint on any error.

**Slumbot, 10,000 hands (2026-07-08), full upgraded stack** (E[HS²]
blueprint + harmonic translation + river re-solving):

```
                              raw:  −152.3 ± 170.8 mbb/hand
                  all-in adjusted:  −249.0 ± 142.7 mbb/hand
AIVAT-lite (all-in + preflop OLS):  −214.4 ± 142.5 mbb/hand
```

**The headline raw improvement (−332 → −152) is mostly all-in luck.**
Luck-adjusted, the upgraded stack (−214 ± 143) and the baseline
(−230 ± 158) are statistically indistinguishable — the baseline ran
~80 mbb unlucky, the upgraded run ~100 mbb lucky, and AIVAT-lite caught
both. This is precisely the failure mode variance reduction exists to
expose: without it we'd have claimed a 180-mbb gain.

Interpretation: at this blueprint scale, bucketing quality, action
translation, and river-only re-solving are not the binding constraints.
The candidates that remain, in rough order of expected impact: much
larger blueprints (Slumbot's trained for months; ours for 10 hours),
finer action abstraction (more bet sizes), re-solving from the flop/turn
rather than river-only, and safe (gadget-game) re-solving so subgame
strategies can't be exploited through a weak blueprint's ranges.

## Per-upgrade ablations (2026-07-08/09, 10,000 hands each, AIVAT-lite)

Each upgrade measured in isolation via `run_ablations.py` (one Slumbot
session at a time, resumable), all figures luck-adjusted mbb/hand:

| Configuration | AIVAT-lite | vs baseline |
|---|---|---|
| baseline (std blueprint, naive, no re-solve) | −230 ± 158 | — |
| E[HS²] only | −260 ± 149 | flat |
| pseudo-harmonic only | −445 ± 144 | ~1.4σ **worse** |
| river re-solving only | **−157 ± 154** | best, ~0.3σ better |
| all three combined | −214 ± 143 | flat |

Nothing is individually significant at 10k hands (per-component SE ≈ 150,
so a real effect needs to clear ~300 mbb). But the *directions* are
consistent and informative: river re-solving is the only component that
leans positive, and it drives the combined stack; harmonic translation
alone leans clearly negative — plausible, since randomizing bet mapping
without a re-solve behind it just adds off-tree exposure the blueprint
can't defend. E[HS²] is a wash. Takeaway: re-solving is the lever with
signal; the next test is whether a richer action set and a bigger
blueprint move the floor (see below).

## Extended action set + bigger blueprint (2026-07-09/10)

`NLHEStateX` adds a 2×-pot overbet (`--actions ext`), roughly doubling the
abstract game. Retrained for **5M iterations** (2× the earlier runs):
`hunl_blueprint_ext.pkl`, 3.0M infosets, E[HS²] buckets. 10,000-hand
Slumbot match:

```
                              raw:  −263.7 ± 177.9 mbb/hand
                  all-in adjusted:  −141.4 ± 154.8 mbb/hand
AIVAT-lite (all-in + preflop OLS):  −119.4 ± 154.6 mbb/hand
```

**Best luck-adjusted result to date** — the first configuration clearly
on the better side of −150. Against the E[HS²]-only ablation (−260, same
buckets, std actions, 2.5M iters) it is +141 mbb better; against the
baseline (−230) it is +111 better. Both comparisons point the same way,
though at ~0.5–0.7σ neither is significant on its own. Honest confound:
this run changed *two* things at once — the overbet action **and** 5M vs
2.5M iterations — so it cannot separate "richer actions" from "more
training." What it does establish: action richness + blueprint scale is
the direction that moves the floor, matching the ablation verdict that
bucketing and translation don't.

## Capstone attempt: ext blueprint + river re-solving — regressed (2026-07-10)

Stacking the two positive-signal levers **failed**, and clearly. Stopped
at 5,023 hands once the trend was unambiguous:

```
                              raw:  −701.9 ± 245.3 mbb/hand
AIVAT-lite (all-in + preflop OLS):  −776.6 ± 196.4 mbb/hand
```

This is a genuine regression, not variance: luck adjustment made it
*worse* (−777 vs −702 raw), so the agent was mildly lucky in all-ins
while still losing −777 of pure skill; and it is 2.6σ below the ext-only
run (−119 ± 155). The OLS beta also collapsed (302 vs the usual ~600),
meaning outcomes decoupled from starting-hand quality — bad postflop
decisions, not bad deals.

### Root cause (two wrong guesses, then the evidence)

Getting this right took two false starts, both recorded here because the
dead ends are instructive:

1. *"The overbet."* Guess: the resolver fires 2×-pot bluffs Slumbot
   punishes. Fix tried: `--resolve-cap-pot` (forbid our own overbets).
   Result: **still −792** at 2,500 hands. Falsified — and it should have
   been obvious that capping *our* bets can't help if the problem is us
   *calling*.
2. *"Range collapse."* A diagnostic (`diag_range.py`) showed the ext
   blueprint's opponent-range estimate collapsing to uniform 97% of the
   time. But that diagnostic had a bug: it rebuilt the shadow with the
   *std* profile, so the trace's 5-action legal sets didn't match the
   ext policy's 6-action vectors, forcing a spurious fallback. Fixed to
   use the blueprint's own profile, the real fallback rate is **0%** and
   the range is healthy (7.7 bits entropy, same as std's 7.6). A
   probability-floor "fix" (`--range-smooth`) was kept as an optional
   defensive knob but is **not** what was wrong.

The decisive step was localizing the loss (`localize.py`) rather than
guessing. Splitting hands by whether we faced a river decision:

| Config | river-decision hands | no river decision |
|---|---|---|
| ext blueprint alone | −601 | −152 |
| std blueprint + resolve | −708 (≈ its own blueprint) | +41 |
| **ext blueprint + resolve** | **−1721** | −365 (control, ≈ noise) |

Re-solving is roughly neutral on the std blueprint but ~−1000 mbb worse
per river hand on the ext blueprint. Inspecting the 12 worst river hands
(`worst_hands.py`) shows the actual mistake, and it is not betting — it is
**calling**: every one is us calling a large bet or all-in with a beaten
hand (ace-high into trip tens for 19,200; one pair into the nut flush on a
four-flush board; KK into a made straight). The resolver systematically
**overcalls**.

Mechanism: unsafe re-solving trusts the blueprint's model of the
opponent's *big-bet* range. The ext blueprint (E[HS²], 6 bet-sizes) can
construct a far more bluff-heavy polarized betting range than the std
profile can, so the solve concludes "call wide" — but Slumbot's big bets
are value-weighted, not bluff-heavy, and it stacks us. The std profile,
with fewer bet-sizes, can't model as bluffy an opponent, so its re-solve
overcalls less. This is the textbook failure of *unsafe* subgame
re-solving; the real fix is **safe re-solving** (a CFR-D / gadget game
that bounds subgame values by the blueprint), which is substantial and
remains future work — not a knob.

**Best validated configuration: the ext blueprint alone, −119 mbb/hand
luck-adjusted. Re-solving does not compose with it until re-solving is
made safe.**

## Safe (gadget) re-solving: implemented, converged, still fails (2026-07-12)

`safe_resolver.py` implements CFR-D-style safe re-solving (Burch et al.
2014): our range from blueprint reach, per-hand counterfactual values
(CBV) from a bucketed best response vs our blueprint, and a gadget where
the opponent chooses Terminate (take CBV) or Follow. The safety property
verifies offline: opponent BR vs the gadget strategy ≤ blueprint CBVs.

Two live runs, one lesson each:

1. **3k gadget iterations (7,524 hands): −2,428 mbb/hand on river hands.**
   Diagnosis (`diag_safe.py`): the two-range gadget is ~10× bigger than
   the unsafe fixed-hand solve, so 3k iterations yields mid-convergence
   noise (77 overbetting 23% where the blueprint checks 97%). Strategies
   stabilize by 10–20k iterations. The offline safety test missed it:
   empty-policy CBVs are generous enough that even noise passes.
2. **15k iterations, converged (2,500-hand validation):**

```
   AIVAT-lite:            −695 ± 300 mbb/hand (luck-adj worse than raw)
   river-decision hands: −2,307 ± 868
   no river decision:      +152 ± 382   (healthy control)
```

Converged or not, river re-solving on the ext blueprint loses ~−2,300 on
river hands vs the blueprint's own −601. Full triangulation:

| River re-solve variant | river-decision hands |
|---|---|
| none (ext blueprint alone) | −601 |
| unsafe | −1,721 |
| safe, undertrained (3k) | −2,428 |
| safe, converged (15k) | −2,307 |

**Conclusion: the failure is in the inputs, and no gadget can launder
them.** All variants share the same range model: opponent hands weighted
by blueprint reach over the abstract-mapped line. Slumbot does not share
our abstraction, so that conditioning is systematically wrong — and the
richer the action set, the sharper (and more wrong) it gets, which is
why std+resolve was ≈neutral while every ext+resolve variant fails. The
safety guarantee is *relative to the CBVs*, which are computed under the
same wrong range: garbage in, guaranteed garbage out. This is exactly
why DeepStack/Libratus carry trunk-consistent ranges and solve-time
values through the whole game instead of re-deriving them at the table.

Retrofit paths (all substantial): trunk re-solving from the hand's start
so ranges stay self-consistent; opponent-model calibration from observed
Slumbot showdowns; or a DeepStack-style value network. At workstation
scale the validated recipe stands: **blueprint scale + action richness,
no decision-time re-solving — ext blueprint alone, −119 mbb/hand.**

## DeepStack-style re-solving + the controlled experiment that reversed the diagnosis (2026-07-13)

`deepstack_resolver.py` implements the paper's input discipline: the
opponent's range is **not an input** (full combo set dealt uniformly,
constrained by CBVs), our range is the only reach estimate (trustworthy
pre-river), Bayes-updated within the hand by our own solved strategies
(continual re-solving). Live validation still failed the pre-registered
kill line (−824 at 1,000 hands; river-decision hands −3,081).

That falsified the "opponent-range input" theory, so a control experiment
was run (`selfplay_ab.py`): the resolved agent vs the pure blueprint
**inside our own engine**, where every model input is correct *by
construction* (the opponent literally plays the blueprint the CBVs are
computed against). Result over 1,000 hands: overall −66 (noise), but
**river-decision hands −4,690 mbb/hand (n=200)**.

**The input-fidelity conclusion was wrong.** A correct safe re-solve can
only refine against the exact modeled opponent; losing in-engine proved
the defect was ours. Score: seven hypotheses instrumented, five falsified
(overbet action, range collapse, opponent-range input, pot drift, code
defect), two confirmed (gadget undertraining, and the root cause below).

### Root cause found: CBV slack (2026-07-13)

The safety guarantee is only as tight as its constraint values. Our CBVs
were **best-response values against a weak blueprint** — enormously loose
(e.g. +3,280 on a 1,600 pot) — so the "safe" region included terrible
strategies, and the gadget equilibrium wandered into them. DeepStack's
constraints are tight because they come from near-equilibrium trunk
solves. Fix: `cbv_mode="blueprint"` computes **self-play continuation
values** (the opponent continues per its own blueprint — what DeepStack's
carried values approximate) instead of BR values. In-engine A/B, same
1,000 hands and seeds:

| CBV mode | river-decision hands |
|---|---|
| br (loose) | −4,690 mbb/hand |
| blueprint (tight) | **−268 mbb/hand** (≈ mirror-zero within noise) |

The gadget was never broken; its constraints were. Live Slumbot
validation of ext + DeepStack-resolve with tight CBVs: in progress.

## Reference results — HULHE ES-MCCFR (30k iterations, 8 buckets, 50 MC samples, seed 0)

```
iter    5000: 49,480 infosets   vs random +1011 mbb/hand   vs call +332 mbb/hand
iter   15000: 51,881 infosets   vs random  +825 mbb/hand   vs call +305 mbb/hand
iter   30000: 52,159 infosets   vs random  +852 mbb/hand   vs call +494 mbb/hand
```

~5 min wall time; policy in `hulhe_policy.pkl`. The abstracted game saturates
at ~52k infosets. The vs-call curve is the informative one (a call-station
cannot be bluffed, so gains there mean better value-betting); 4000-hand evals
carry ~±40 mbb sampling noise, which explains the vs-random wobble. For
calibration, top HULHE bots beat weak players by several hundred mbb/hand;
Cepheus-level play is ≤ ~1 mbb/hand exploitable.

## Reference results — Leduc exploitability (chips/hand, seed 0)

```
algorithm  iterations  exploitability  wall time
cfr             200        0.0538        16.4s
cfr+            200        0.0050        18.5s
mccfr        20,000        0.1831         7.1s   (external sampling)
```

CFR+ is ~11x tighter than vanilla CFR at equal iteration count, matching the
literature. MCCFR's iterations are ~450x cheaper but each carries sampling
noise; it wins when the full tree is too big to traverse (rungs 3+), not here.

## Key sources

- Neller & Lanctot, *An Introduction to Counterfactual Regret Minimization*
- Brown & Sandholm, *Libratus* (Science 2018), *Pluribus* (Science 2019)
- Moravčík et al., *DeepStack* (Science 2017)
- RLCard: https://github.com/datamllab/rlcard · OpenSpiel: https://github.com/deepmind/open_spiel
- Benchmark opponent: Slumbot (https://www.slumbot.com)

**Scope note:** research/benchmark use only — deploying bots on real-money
sites violates their ToS and may be illegal in some jurisdictions.
