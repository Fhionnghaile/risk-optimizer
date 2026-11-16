"""Produce the efficient frontier, compare against baselines, and write artifacts:
  examples/out/frontier.csv   — the exact Pareto frontier (spend, residual mean loss, tail)
  examples/out/frontier.png   — the frontier vs baselines, with tail-risk overlay
  examples/out/summary.md     — the optimal portfolio at a chosen budget + headline numbers
"""
import os, csv, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from catalog import build
from riskopt.optimize import (pareto_frontier, min_loss_at_budget,
                              greedy_by_roi, greedy_by_efficacy, random_portfolio_loss)

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def tail_for(model, controls, q=0.95):
    losses = model.simulate(controls, iters=120_000, seed=11)
    return model.var_cvar(losses, q)


def main():
    m = build()
    base = m.expected_annual_loss(())
    base_var, base_cvar = tail_for(m, ())
    total_cost = m.portfolio_cost([c.key for c in m.controls])

    fr = pareto_frontier(m)

    # budget grid for baseline comparison
    grid = np.linspace(0, total_cost, 41)

    # write frontier.csv (exact optimum at each grid budget + tail)
    rows = []
    for b in grid:
        pt = min_loss_at_budget(m, b)
        var, cvar = tail_for(m, pt.controls)
        rows.append((b, pt.cost, pt.residual_loss, var, cvar, "|".join(pt.controls)))
    with open(os.path.join(OUT, "frontier.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["budget", "spend", "residual_expected_loss", "residual_var95", "residual_cvar95", "controls"])
        w.writerows(rows)

    # baselines at each budget
    opt = [min_loss_at_budget(m, b).residual_loss for b in grid]
    g_roi = [greedy_by_roi(m, b).residual_loss for b in grid]
    g_eff = [greedy_by_efficacy(m, b).residual_loss for b in grid]
    rnd = [random_portfolio_loss(m, b, trials=1500) for b in grid]

    # --- chart (colourblind-safe, one axis in £, direct-labelled) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = 1000.0
    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=150)
    ax.plot(grid/k, np.array(opt)/k,   color="#1f6feb", lw=2.4, label="Optimal (exact frontier)", zorder=5)
    ax.plot(grid/k, np.array(g_roi)/k, color="#d08c00", lw=1.8, ls="--", label="Greedy by ROI")
    ax.plot(grid/k, np.array(g_eff)/k, color="#8957e5", lw=1.8, ls=":",  label="Greedy by efficacy")
    ax.plot(grid/k, np.array(rnd)/k,   color="#999999", lw=1.4, ls="-.", label="Best random (1500)")
    ax.scatter([0],[base/k], color="#cf222e", zorder=6)
    ax.annotate(f"no controls\n£{base/k:,.0f}k expected loss", (0, base/k),
                textcoords="offset points", xytext=(12, -4), fontsize=9, color="#cf222e")
    ax.set_xlabel("Annual control spend (£k)")
    ax.set_ylabel("Residual expected annual loss (£k)")
    ax.set_title("Security-investment efficient frontier: spend vs residual risk")
    ax.grid(True, color="#e6e6e6", lw=0.8)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "frontier.png"))

    # --- summary at an illustrative budget ---
    budget = 400_000
    pt = min_loss_at_budget(m, budget)
    var, cvar = tail_for(m, pt.controls)
    names = [next(c.name for c in m.controls if c.key==k_) for k_ in pt.controls]
    red = 100*(1 - pt.residual_loss/base)
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("# Efficient-frontier result (worked example)\n\n")
        f.write(f"- Baseline expected annual loss (no controls): **£{base:,.0f}**\n")
        f.write(f"- Baseline 95% CVaR (tail): **£{base_cvar:,.0f}**\n")
        f.write(f"- Full catalog cost: £{total_cost:,.0f}\n\n")
        f.write(f"## Optimal portfolio at a £{budget:,.0f} budget\n\n")
        f.write(f"- Spend: **£{pt.cost:,.0f}**\n")
        f.write(f"- Residual expected annual loss: **£{pt.residual_loss:,.0f}** "
                f"(**{red:.0f}%** reduction)\n")
        f.write(f"- Residual 95% CVaR: **£{cvar:,.0f}**\n")
        f.write(f"- Controls chosen: {', '.join(names)}\n\n")
        # how much worse the heuristics are at this budget
        f.write("## Why 'obvious' selection underperforms (same budget)\n\n")
        for label, fn in [("Greedy by ROI", greedy_by_roi), ("Greedy by efficacy", greedy_by_efficacy)]:
            hp = fn(m, budget)
            gap = hp.residual_loss - pt.residual_loss
            f.write(f"- {label}: residual £{hp.residual_loss:,.0f} "
                    f"(**£{gap:,.0f}/yr worse** than optimal at equal budget)\n")
    print("wrote frontier.csv, frontier.png, summary.md")
    print(open(os.path.join(OUT, "summary.md")).read())


if __name__ == "__main__":
    main()
