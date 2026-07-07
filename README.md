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

Evaluation note: there is no 6-max Slumbot, so the metric is mbb/hand vs
agent pools with the hero rotated through every seat. With 3+ players CFR
keeps no Nash guarantee (only dominated-strategy elimination), so pool
results *are* the success criterion, not a proxy for exploitability.

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

10k-hand match of the full upgraded stack vs Slumbot: in progress.

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
