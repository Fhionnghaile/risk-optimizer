import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import numpy as np
import pytest
from riskopt.model import Scenario, Control, RiskModel
from riskopt.optimize import (pareto_frontier, min_loss_at_budget, bnb_min_loss,
                              greedy_by_roi, greedy_by_efficacy, random_portfolio_loss)
from catalog import build


# ---- model correctness ----------------------------------------------------
def test_wald_mean_matches_monte_carlo():
    """Closed-form E[S]=lef*lm_mean must match the compound-Poisson simulation."""
    m = RiskModel([Scenario("x", "x", lef=0.8, lm_mean=500_000, lm_cv=1.5)], [])
    analytic = m.expected_annual_loss(())
    sim = m.simulate((), iters=400_000, seed=3).mean()
    assert abs(sim - analytic) / analytic < 0.02           # within 2% MC error


def test_controls_compose_multiplicatively():
    s = Scenario("x", "x", lef=1.0, lm_mean=100.0, lm_cv=1.0)
    c1 = Control("a", "a", 1.0, freq_reduction={"x": 0.5})
    c2 = Control("b", "b", 1.0, freq_reduction={"x": 0.5})
    m = RiskModel([s], [c1, c2])
    assert m.expected_annual_loss(()) == pytest.approx(100.0)
    assert m.expected_annual_loss(["a"]) == pytest.approx(50.0)
    assert m.expected_annual_loss(["a", "b"]) == pytest.approx(25.0)   # 0.5*0.5, not 0


def test_reduction_bounds_enforced():
    with pytest.raises(ValueError):
        Control("a", "a", 1.0, freq_reduction={"x": 1.0})


# ---- optimiser exactness --------------------------------------------------
def test_knapsack_tiny_hand_checkable():
    """Three controls, hand-computable optimum under a tight budget."""
    s = Scenario("x", "x", lef=1.0, lm_mean=1000.0, lm_cv=1.0)
    cs = [Control("cheap", "cheap", 10, freq_reduction={"x": 0.3}),
          Control("mid", "mid", 20, freq_reduction={"x": 0.5}),
          Control("dear", "dear", 25, freq_reduction={"x": 0.6})]
    m = RiskModel([s], cs)
    # budget 30: options within budget: {}, cheap, mid, dear, cheap+mid(30), cheap+dear(35 no), mid+dear(45 no)
    # losses: 1000, 700, 500, 400, 350 -> best is dear alone (400)? cheap+mid=1000*0.7*0.5=350 < 400
    pt = min_loss_at_budget(m, 30)
    assert pt.residual_loss == pytest.approx(350.0)
    assert set(pt.controls) == {"cheap", "mid"}


def test_bnb_equals_bruteforce_on_catalog():
    m = build()
    for b in (0, 100_000, 250_000, 400_000, 700_000, 1_000_000):
        assert bnb_min_loss(m, b) == pytest.approx(min_loss_at_budget(m, b).residual_loss, rel=1e-6)


def test_frontier_is_monotone_nonincreasing():
    m = build()
    fr = pareto_frontier(m)
    costs = [p.cost for p in fr]
    losses = [p.residual_loss for p in fr]
    assert costs == sorted(costs)
    # strictly decreasing residual loss as spend increases along the frontier
    assert all(losses[i] > losses[i+1] for i in range(len(losses)-1))


def test_optimal_dominates_every_baseline():
    """The exact optimum is <= every heuristic at every budget (the whole point)."""
    m = build()
    total = m.portfolio_cost([c.key for c in m.controls])
    strict = 0
    for b in np.linspace(0, total, 21):
        opt = min_loss_at_budget(m, b).residual_loss
        for fn in (greedy_by_roi, greedy_by_efficacy):
            h = fn(m, b).residual_loss
            assert opt <= h + 1e-6
            if h > opt + 1.0:
                strict += 1
        assert opt <= random_portfolio_loss(m, b, trials=800) + 1e-6
    # and somewhere a heuristic is strictly, materially worse (else the point is moot)
    assert strict > 0


def test_budget_zero_is_baseline():
    m = build()
    assert min_loss_at_budget(m, 0).residual_loss == pytest.approx(m.expected_annual_loss(()))
