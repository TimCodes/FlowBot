"""Rung 5.6: online opponent tendency model for the exploitation layer.

Per opponent identity, per (street, facing-a-bet) context, the model counts
observed fold / call / raise action classes from the public history of
completed hands (no hole-card information is used). At decision time it
reshapes a policy toward the observed class frequencies with a capped
mixture:

    modeled = (1 - lam) * policy + lam * observed_class_freqs
    lam     = min(lam_max, n / (n + prior_hands))

Class mass is spread across a class's concrete actions (e.g. raise mass
over h/p/a) proportionally to the base policy, so within-class shape is
preserved. Properties this buys:

  * No data => lam = 0 => exactly the base policy: adding the model can
    never change behavior against unobserved opponents (regression-safe).
  * A call-station drives lam -> lam_max with call freq 1.0 in facing
    contexts: modeled ranges widen and modeled continuation play stops
    folding, so the search stops bluffing and value-bets thinner.
  * lam_max < 1 keeps a restricted-Nash-response-style hedge: the model
    never fully overrides the equilibrium prior, bounding how badly a
    wrong model can mislead the search.

Contexts are keyed by facing/not-facing so the fold class only ever carries
mass where folding is legal. Classes absent from the current legal set
(e.g. no raises when facing an all-in) lose their mass to renormalization.
"""

from __future__ import annotations

from nlhe6_engine import CALL, FOLD

CLASSES = ("fold", "call", "raise")


def action_class(action: str) -> str:
    if action == FOLD:
        return "fold"
    if action == CALL:
        return "call"
    return "raise"


class TendencyModel:
    def __init__(self, prior_hands: float = 30.0, lam_max: float = 0.8):
        self.prior_hands = prior_hands
        self.lam_max = lam_max
        # (identity, street, facing) -> {class: count}
        self.counts: dict[tuple, dict[str, float]] = {}

    def observe_decisions(self, decisions, identities):
        """Ingest (state_before, action) pairs from a completed hand's public
        history (see nlhe6_search.replay_decisions). `identities[seat]` is a
        stable opponent id, or None for seats not to be modeled (the hero)."""
        for st, ch in decisions:
            seat = st.to_act
            ident = identities[seat]
            if ident is None:
                continue
            facing = int(max(st.contrib) > st.contrib[seat])
            key = (ident, st.street, facing)
            ctx = self.counts.setdefault(key, dict.fromkeys(CLASSES, 0.0))
            ctx[action_class(ch)] += 1.0

    def observations(self, ident) -> float:
        return sum(sum(ctx.values()) for key, ctx in self.counts.items()
                   if key[0] == ident)

    def modulate(self, probs, actions, ident, street, facing):
        """Reshape `probs` (over `actions`) toward observed tendencies."""
        ctx = self.counts.get((ident, street, facing))
        if not ctx:
            return probs
        n = sum(ctx.values())
        if n <= 0:
            return probs
        lam = min(self.lam_max, n / (n + self.prior_hands))
        class_mass = dict.fromkeys(CLASSES, 0.0)
        for p, a in zip(probs, actions):
            class_mass[action_class(a)] += p
        out = []
        for p, a in zip(probs, actions):
            c = action_class(a)
            if class_mass[c] > 0:
                target = (ctx[c] / n) * (p / class_mass[c])
            else:
                target = 0.0
            out.append((1.0 - lam) * p + lam * target)
        total = sum(out)
        if total <= 0:
            return probs
        return [x / total for x in out]

    def modulator(self, ident):
        """Bind an identity into the callable shape SubgameSolver accepts."""
        def mod(probs, actions, street, facing):
            return self.modulate(probs, actions, ident, street, facing)
        return mod
