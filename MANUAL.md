# risk-optimizer — the efficient frontier of security spend — manual

*One page, eight blocks. Every figure here comes from this repository's own tests and README.*

## 1. Claim
At a £400k budget the exact optimum cuts expected loss 71%, to £520k, and greedy-by-efficacy lands £313k a year worse at the same spend (illustrative catalogue).

## 2. What it is
Given loss scenarios and a catalogue of controls with costs, this tool computes the provably best set to buy at every budget, and shows how far common rules of thumb fall short.

## 3. Demo
Frontier chart: `docs/frontier.png`. `python examples/run.py` regenerates `examples/out/frontier.csv`, `frontier.png` and `summary.md`.

## 4. Run it yourself
**Prerequisites:** Python 3.

```bash
pip install -e .[test]
pytest -q
python examples/run.py
```

## 5. What you will see
8 tests pass, including that branch-and-bound equals brute force, the frontier is monotone and the optimum dominates every baseline.

## 6. 30-second talking point
> "Are we spending the right amount, on the right things?" deserves an optimum, not a heat-map. Because control benefits have diminishing returns, greedy picks leave money on the table, and this shows exactly how much.

## 7. Scope and limits
- Example frequencies, magnitudes and efficacies are illustrative defaults to be calibrated to real loss data.
- Controls are modelled as independent; cross-control synergies are a documented simplification.
- Exhaustive enumeration is exact up to about 20 controls. Beyond that, use the branch-and-bound or an ILP.

## 8. Links
[Repository](https://github.com/fhionnghaile/risk-optimizer) · [Write-up](https://padraigfhionnghaile.com/work/risk-optimizer/) · [README](README.md)
