# Financial Contagion Lab

A network stress-testing engine implementing Eisenberg–Noe fixed-point clearing, endogenous defaults, payment shortfalls, and marginal systemic-impact rankings.

```python
from model import clear, rank_systemic_nodes
result = clear(liability_matrix, external_assets, outside_liabilities, shock=asset_haircuts)
ranking = rank_systemic_nodes(liability_matrix, external_assets)
```

The model asks a useful counterfactual: **which institution creates the largest additional network shortfall when its external assets are shocked?** The answer can differ from balance-sheet size because network position changes how losses propagate.

It also asks a second, decision-focused counterfactual: **if every possible
capital placement costs the same, where does one unit prevent the most total
shortfall after a specified shock?** `rank_capital_interventions` clears the
entire network again for every candidate placement. The result separates the
institution most likely to transmit damage from the institution where limited
support absorbs the most damage; neither ranking has to match institution size.

## Distinctive edge

The contribution is a transparent shock-to-action loop: fixed-point clearing
reveals endogenous defaults, node counterfactuals rank marginal transmitters,
and equal-budget intervention counterfactuals rank where capital prevents the
most system-wide shortfall. This moves the model from describing a cascade to
testing a concrete mitigation decision under the same network assumptions.

## Boundaries

This is a stylized research model. Liability data are rarely complete; recovery, netting, collateral, liquidity, fire sales, timing, and policy intervention require richer treatment. Results are scenario outputs, not forecasts or financial advice.

Requires NumPy. Tests use `pytest -q`.
