"""Eisenberg–Noe clearing and marginal systemic-impact experiments."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ClearingResult:
    payments: np.ndarray
    shortfall: np.ndarray
    defaults: np.ndarray
    iterations: int


def clear(liabilities, external_assets, outside_liabilities=None, shock=None, recovery=1.0, tolerance=1e-10, max_iter=10_000):
    """Compute the greatest fixed-point clearing vector for a liability network."""
    L, assets = np.asarray(liabilities, float), np.asarray(external_assets, float)
    if L.ndim != 2 or L.shape[0] != L.shape[1] or assets.size != L.shape[0]:
        raise ValueError("liabilities must be square with one asset value per institution")
    outside = np.zeros_like(assets) if outside_liabilities is None else np.asarray(outside_liabilities, float)
    if outside.size != assets.size:
        raise ValueError("outside_liabilities must have one value per institution")
    total = L.sum(axis=1) + outside
    relative = np.divide(L, total[:, None], out=np.zeros_like(L), where=total[:, None] > 0)
    shocked = assets * (1 - (np.zeros_like(assets) if shock is None else np.asarray(shock, float)))
    payments = total.copy()
    for iteration in range(1, max_iter + 1):
        incoming = relative.T @ payments
        updated = np.minimum(total, np.maximum(0, shocked + recovery * incoming))
        if np.max(np.abs(updated - payments)) < tolerance:
            payments = updated
            break
        payments = updated
    shortfall = total - payments
    return ClearingResult(payments, shortfall, shortfall > tolerance, iteration)


def marginal_systemic_impact(liabilities, assets, institution, shock_size=0.25, outside_liabilities=None):
    baseline = clear(liabilities, assets, outside_liabilities)
    shock = np.zeros(len(assets)); shock[institution] = shock_size
    stressed = clear(liabilities, assets, outside_liabilities, shock)
    return float(stressed.shortfall.sum() - baseline.shortfall.sum())


def rank_systemic_nodes(liabilities, assets, shock_size=0.25, outside_liabilities=None):
    impacts = [marginal_systemic_impact(liabilities, assets, i, shock_size, outside_liabilities) for i in range(len(assets))]
    return sorted(enumerate(impacts), key=lambda item: item[1], reverse=True)
