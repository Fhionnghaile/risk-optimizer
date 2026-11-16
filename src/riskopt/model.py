"""FAIR-style risk model with controllable scenarios.

A *scenario* is a loss-event class modelled in FAIR terms:
  - lef    : Loss Event Frequency (events/year), the rate of a compound-Poisson process
  - lm_mean: mean single-loss magnitude (currency)
  - lm_cv  : coefficient of variation of single-loss magnitude (for the tail model)

Annual loss for a scenario is a compound Poisson sum:  S = sum_{i=1..N} X_i,
with N ~ Poisson(lef) and X_i a lognormal with the given mean and CV.

Two facts we rely on (both exact, not approximations):
  E[S]   = lef * lm_mean                      (Wald's identity)
  Var[S] = lef * E[X^2] = lef * lm_mean^2 * (1 + cv^2)   (compound-Poisson variance)

So the *mean* annual loss is available in closed form — no Monte-Carlo needed to
optimise against it. The *tail* (VaR/CVaR) needs simulation, which `simulate()` does
with common random numbers so portfolios are compared on the same sampled world.

A *control* reduces a scenario's frequency and/or magnitude multiplicatively. Two
controls acting on the same scenario compose as the product of their residual factors
(independent, order-free) — which makes the benefit of a control-set *submodular*
(diminishing returns), the property that defeats naive greedy selection.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import math
import numpy as np


@dataclass(frozen=True)
class Scenario:
    key: str
    name: str
    lef: float          # loss events / year
    lm_mean: float      # mean single-loss magnitude (currency)
    lm_cv: float = 1.5  # coefficient of variation of single-loss magnitude

    def __post_init__(self):
        if self.lef < 0 or self.lm_mean < 0 or self.lm_cv < 0:
            raise ValueError("scenario parameters must be non-negative")


@dataclass(frozen=True)
class Control:
    key: str
    name: str
    cost: float                      # annual cost (currency)
    freq_reduction: Dict[str, float] = field(default_factory=dict)  # scenario_key -> fraction in [0,1)
    mag_reduction: Dict[str, float] = field(default_factory=dict)   # scenario_key -> fraction in [0,1)

    def __post_init__(self):
        if self.cost < 0:
            raise ValueError("cost must be non-negative")
        for d in (self.freq_reduction, self.mag_reduction):
            for v in d.values():
                if not (0.0 <= v < 1.0):
                    raise ValueError("reductions must be in [0,1)")


class RiskModel:
    """A portfolio of scenarios plus a catalog of candidate controls."""

    def __init__(self, scenarios: List[Scenario], controls: List[Control]):
        self.scenarios = list(scenarios)
        self.controls = list(controls)
        self._skeys = [s.key for s in self.scenarios]
        ck = [c.key for c in self.controls]
        if len(set(self._skeys)) != len(self._skeys):
            raise ValueError("duplicate scenario keys")
        if len(set(ck)) != len(ck):
            raise ValueError("duplicate control keys")

    # ---- residual parameters after applying a set of controls -----------------
    def residual_params(self, control_keys):
        """Return {scenario_key: (residual_lef, residual_lm_mean)} after controls."""
        chosen = [c for c in self.controls if c.key in set(control_keys)]
        out = {}
        for s in self.scenarios:
            ff = 1.0
            mf = 1.0
            for c in chosen:
                ff *= (1.0 - c.freq_reduction.get(s.key, 0.0))
                mf *= (1.0 - c.mag_reduction.get(s.key, 0.0))
            out[s.key] = (s.lef * ff, s.lm_mean * mf)
        return out

    # ---- exact mean annual loss (Wald) ---------------------------------------
    def expected_annual_loss(self, control_keys=()) -> float:
        rp = self.residual_params(control_keys)
        return float(sum(lef * lm for (lef, lm) in rp.values()))

    def portfolio_cost(self, control_keys) -> float:
        ck = set(control_keys)
        return float(sum(c.cost for c in self.controls if c.key in ck))

    # ---- Monte-Carlo for the tail (VaR / CVaR) -------------------------------
    def simulate(self, control_keys=(), iters=200_000, seed=7):
        """Simulate the annual-loss distribution (compound Poisson of lognormals).

        Common random numbers via the fixed seed, so two portfolios are evaluated on
        the same sampled world and differences are not Monte-Carlo noise.
        """
        rng = np.random.default_rng(seed)
        rp = self.residual_params(control_keys)
        total = np.zeros(iters)
        for s in self.scenarios:
            lef, lm_mean = rp[s.key]
            if lef <= 0 or lm_mean <= 0:
                continue
            counts = rng.poisson(lef, size=iters)
            cap = int(counts.max()) if counts.size else 0
            if cap == 0:
                continue
            # lognormal params from mean and CV:  sigma^2 = ln(1+cv^2)
            sigma = math.sqrt(math.log(1.0 + s.lm_cv ** 2))
            mu = math.log(lm_mean) - 0.5 * sigma ** 2
            # draw a (iters x cap) matrix once, mask by per-row event count
            draws = rng.lognormal(mu, sigma, size=(iters, cap))
            idx = np.arange(cap)
            mask = idx[None, :] < counts[:, None]
            total += (draws * mask).sum(axis=1)
        return total

    @staticmethod
    def var_cvar(losses, q=0.95):
        """Value-at-Risk and Conditional VaR (expected shortfall) at quantile q."""
        v = float(np.quantile(losses, q))
        tail = losses[losses >= v]
        c = float(tail.mean()) if tail.size else v
        return v, c
