import numpy as np
from model import clear, rank_capital_interventions, rank_systemic_nodes


def test_solvent_network_clears_in_full():
    liabilities = np.array([[0, 30, 0], [0, 0, 20], [10, 0, 0]], float)
    result = clear(liabilities, np.array([50, 50, 50]))
    assert not result.defaults.any()
    assert np.allclose(result.shortfall, 0)


def test_stress_can_propagate_and_ranking_is_complete():
    liabilities = np.array([[0, 80, 0], [0, 0, 80], [80, 0, 0]], float)
    assets = np.array([15, 15, 15], float)
    outside = np.array([20, 20, 20], float)
    stressed = clear(liabilities, assets, outside, shock=np.array([1, 0, 0]))
    assert stressed.shortfall.sum() > 0
    assert len(rank_systemic_nodes(liabilities, assets, outside_liabilities=outside)) == 3


def test_equal_cost_interventions_are_ranked_by_prevented_shortfall():
    liabilities = np.array([[0, 75, 0], [0, 0, 60], [0, 0, 0]], float)
    assets = np.array([10, 5, 5], float)
    outside = np.array([20, 20, 20], float)
    shock = np.array([1.0, 0.0, 0.0])
    ranking = rank_capital_interventions(
        liabilities, assets, shock, capital=10, outside_liabilities=outside
    )
    assert len(ranking) == 3
    assert ranking[0][1] >= ranking[-1][1]
    assert ranking[0][1] > 0


def test_intervention_budget_must_be_positive():
    liabilities = np.zeros((2, 2))
    try:
        rank_capital_interventions(liabilities, np.ones(2), np.zeros(2), capital=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("expected a positive-budget validation error")
