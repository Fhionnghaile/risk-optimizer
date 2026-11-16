"""Portfolio optimisation: minimise residual expected annual loss subject to a budget,
and trace the efficient frontier (spend vs residual risk).

Why the objective is submodular, and why that matters
-----------------------------------------------------
Each control scales a scenario's loss multiplicatively, so the marginal benefit of a
control shrinks as other controls that touch the same scenarios are added. The set
function "reduction in expected loss" is therefore submodular, and greedy selection is
only a (1 - 1/e) approximation — it can and does leave money on the table. We solve the
problem *exactly* so we can show precisely how far the heuristics fall short.

Exactness
---------
`exact_frontier` enumerates all 2^n control subsets (n small by construction for a
reviewable catalog) and keeps, for every achievable spend, the minimum residual loss —
i.e. the true lower envelope. For larger catalogs, `bnb_min_loss` is a branch-and-bound
that prunes with an admissible bound; a test checks it equals brute force.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple
import numpy as np

from .model import RiskModel


@dataclass(frozen=True)
class FrontierPoint:
    budget: float          # the spend cap this point answers
    cost: float            # actual spend of the optimal portfolio (<= budget)
    residual_loss: float   # minimal residual expected annual loss at this spend
    controls: Tuple[str, ...]


def _all_portfolios(model: RiskModel):
    keys = [c.key for c in model.controls]
    n = len(keys)
    for mask in range(1 << n):
        chosen = tuple(keys[i] for i in range(n) if mask & (1 << i))
        yield chosen


def pareto_frontier(model: RiskModel) -> List[FrontierPoint]:
    """Exact Pareto frontier of (spend, residual expected loss).

    Returns points sorted by spend; each is the cheapest way to reach its residual-loss
    level, and no cheaper portfolio achieves a lower loss (the efficient frontier).
    """
    pts = []
    for chosen in _all_portfolios(model):
        cost = model.portfolio_cost(chosen)
        loss = model.expected_annual_loss(chosen)
        pts.append((cost, loss, chosen))
    # for each distinct cost keep the min loss, then keep only non-dominated points
    pts.sort(key=lambda t: (t[0], t[1]))
    frontier: List[FrontierPoint] = []
    best_loss = float("inf")
    for cost, loss, chosen in pts:
        if loss < best_loss - 1e-9:          # strictly better loss than anything cheaper
            best_loss = loss
            frontier.append(FrontierPoint(cost, cost, loss, chosen))
    return frontier


def min_loss_at_budget(model: RiskModel, budget: float) -> FrontierPoint:
    """Exact minimum residual expected loss achievable for spend <= budget."""
    best = None
    for chosen in _all_portfolios(model):
        cost = model.portfolio_cost(chosen)
        if cost > budget + 1e-9:
            continue
        loss = model.expected_annual_loss(chosen)
        if best is None or loss < best[0] - 1e-12 or (abs(loss - best[0]) < 1e-9 and cost < best[1]):
            best = (loss, cost, chosen)
    loss, cost, chosen = best
    return FrontierPoint(budget, cost, loss, tuple(chosen))


# --------------------------------------------------------------------------- #
#  Baselines (what practitioners actually do) — to show the optimiser dominates
# --------------------------------------------------------------------------- #
def greedy_by_roi(model: RiskModel, budget: float) -> FrontierPoint:
    """Repeatedly add the control with the best marginal loss-reduction per £."""
    chosen: List[str] = []
    remaining = {c.key for c in model.controls}
    spent = 0.0
    while True:
        base = model.expected_annual_loss(chosen)
        best = None
        for k in remaining:
            c = next(x for x in model.controls if x.key == k)
            if spent + c.cost > budget + 1e-9 or c.cost <= 0:
                continue
            gain = base - model.expected_annual_loss(chosen + [k])
            roi = gain / c.cost
            if best is None or roi > best[0]:
                best = (roi, k, c.cost)
        if best is None:
            break
        _, k, cost = best
        chosen.append(k); remaining.discard(k); spent += cost
    return FrontierPoint(budget, model.portfolio_cost(chosen),
                         model.expected_annual_loss(chosen), tuple(chosen))


def greedy_by_efficacy(model: RiskModel, budget: float) -> FrontierPoint:
    """Add the biggest absolute loss-reducer first, ignoring cost (a common mistake)."""
    chosen: List[str] = []
    remaining = {c.key for c in model.controls}
    spent = 0.0
    while True:
        base = model.expected_annual_loss(chosen)
        best = None
        for k in remaining:
            c = next(x for x in model.controls if x.key == k)
            if spent + c.cost > budget + 1e-9:
                continue
            gain = base - model.expected_annual_loss(chosen + [k])
            if best is None or gain > best[0]:
                best = (gain, k, c.cost)
        if best is None:
            break
        _, k, cost = best
        chosen.append(k); remaining.discard(k); spent += cost
    return FrontierPoint(budget, model.portfolio_cost(chosen),
                         model.expected_annual_loss(chosen), tuple(chosen))


def random_portfolio_loss(model: RiskModel, budget: float, trials=2000, seed=1) -> float:
    """Best residual loss found by sampling budget-feasible random portfolios."""
    rng = np.random.default_rng(seed)
    keys = [c.key for c in model.controls]
    best = model.expected_annual_loss(())
    for _ in range(trials):
        mask = rng.integers(0, 2, size=len(keys)).astype(bool)
        chosen = [keys[i] for i in range(len(keys)) if mask[i]]
        if model.portfolio_cost(chosen) <= budget + 1e-9:
            best = min(best, model.expected_annual_loss(chosen))
    return best


def bnb_min_loss(model: RiskModel, budget: float) -> float:
    """Branch-and-bound exact min residual loss for spend <= budget (for larger n).

    Admissible lower bound: apply *all* still-undecided affordable controls at once
    (their combined multiplicative reduction) ignoring the budget — no real selection can
    beat that, so it is a valid optimistic bound to prune against.
    """
    controls = model.controls
    n = len(controls)
    order = sorted(range(n), key=lambda i: controls[i].cost)
    best = [model.expected_annual_loss(())]

    def bound(chosen_keys, undecided_idx, spent):
        affordable = [controls[i].key for i in undecided_idx
                      if controls[i].cost <= budget - spent + 1e-9]
        return model.expected_annual_loss(list(chosen_keys) + affordable)

    def rec(pos, chosen, spent):
        if pos == len(order):
            best[0] = min(best[0], model.expected_annual_loss(chosen))
            return
        remaining = order[pos:]
        if bound(chosen, remaining, spent) >= best[0] - 1e-9:
            # even the optimistic all-in bound can't beat current best -> prune
            cur = model.expected_annual_loss(chosen)
            best[0] = min(best[0], cur)
            return
        i = order[pos]
        c = controls[i]
        # branch: include (if affordable)
        if spent + c.cost <= budget + 1e-9:
            rec(pos + 1, chosen + [c.key], spent + c.cost)
        # branch: exclude
        rec(pos + 1, chosen, spent)

    rec(0, [], 0.0)
    return best[0]
