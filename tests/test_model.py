import numpy as np
from model import clear, rank_systemic_nodes


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
