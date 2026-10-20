# Financial Contagion Lab

A network stress-testing engine implementing Eisenberg–Noe fixed-point clearing, endogenous defaults, payment shortfalls, and marginal systemic-impact rankings.

```python
from model import clear, rank_systemic_nodes
result = clear(liability_matrix, external_assets, outside_liabilities, shock=asset_haircuts)
ranking = rank_systemic_nodes(liability_matrix, external_assets)
```

The model asks a useful counterfactual: **which institution creates the largest additional network shortfall when its external assets are shocked?** The answer can differ from balance-sheet size because network position changes how losses propagate.

## Boundaries

This is a stylized research model. Liability data are rarely complete; recovery, netting, collateral, liquidity, fire sales, timing, and policy intervention require richer treatment. Results are scenario outputs, not forecasts or financial advice.

Requires NumPy. Tests use `pytest -q`.
