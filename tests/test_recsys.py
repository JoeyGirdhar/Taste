"""Metric checks and a test that MF can recover a low rank matrix."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from recsys.metrics import rmse, mae
from recsys import MatrixFactorization


def test_metrics_known_values():
    pred = np.array([3.0, 4.0, 2.0])
    actual = np.array([2.0, 4.0, 4.0])
    # errors 1, 0, -2: MAE 1.0, RMSE sqrt(5/3)
    assert abs(mae(pred, actual) - 1.0) < 1e-9
    assert abs(rmse(pred, actual) - np.sqrt(5 / 3)) < 1e-9
    print("[metrics] known-value check OK")


def test_mf_can_learn():
    """Build ratings from true latent factors; MF should recover them (low error)."""
    rng = np.random.default_rng(0)
    n_users, n_items, k = 40, 30, 3
    P = rng.normal(0, 1, (n_users, k))
    Q = rng.normal(0, 1, (n_items, k))
    true = 3.0 + P @ Q.T

    u, i, r = [], [], []
    for uu in range(n_users):
        for ii in range(n_items):
            u.append(uu); i.append(ii); r.append(true[uu, ii])
    train = (np.array(u), np.array(i), np.array(r))

    model = MatrixFactorization(n_factors=3, lr=0.02, reg=0.0, epochs=60, seed=1)
    model.fit(train, n_users, n_items)
    train_rmse = rmse(model.predict(train[0], train[1]), train[2])
    assert train_rmse < 0.15, f"MF failed to fit synthetic data (RMSE {train_rmse:.3f})"
    print(f"[mf] recovered latent factors, train RMSE {train_rmse:.3f}  OK")


if __name__ == "__main__":
    test_metrics_known_values()
    test_mf_can_learn()
    print("\nAll recsys tests passed.")
