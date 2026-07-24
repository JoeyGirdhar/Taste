"""
Train every model and print one honest comparison table.

This is the "measured, not vibes" centerpiece: same train/test split for all
models, same metrics, side by side. It makes the story obvious -- personalization
(matrix factorization) beats knowing-nothing (global mean) by a wide margin, and
you can see exactly how much each idea buys you.

Run:  python examples/evaluate_all.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from recsys import MovieLens, GlobalMean, BiasModel, ItemCF, MatrixFactorization
from recsys.metrics import rmse, mae, precision_recall_at_k


def main():
    data = MovieLens(test_frac=0.2, seed=0)
    print(f"MovieLens 100K -> {data.n_users} users, {data.n_items} movies")
    print(f"train: {len(data.train[2])} ratings   test: {len(data.test[2])} ratings\n")

    models = {
        "Global Mean": GlobalMean(),
        "Bias (mu+b_u+b_i)": BiasModel(epochs=20),
        "Item-Item CF": ItemCF(k_neighbors=40),
        "Matrix Factorization": MatrixFactorization(n_factors=40, epochs=25),
    }

    rows = []
    for name, model in models.items():
        t0 = time.time()
        model.fit(data.train, data.n_users, data.n_items)
        pred = model.predict(data.test[0], data.test[1])
        r = rmse(pred, data.test[2])
        m = mae(pred, data.test[2])
        # Ranking metrics are expensive for CF; skip the trivial Global Mean.
        if name == "Global Mean":
            p_at_k = rec_at_k = float("nan")
        else:
            p_at_k, rec_at_k = precision_recall_at_k(
                model, data.train, data.test, data.n_items, k=10
            )
        rows.append((name, r, m, p_at_k, rec_at_k, time.time() - t0))
        print(f"  {name:22s} done in {time.time() - t0:5.1f}s")

    print("\n" + "=" * 78)
    print(f"{'Model':22s} {'RMSE':>7s} {'MAE':>7s} {'P@10':>7s} {'R@10':>7s} {'train(s)':>9s}")
    print("-" * 78)
    for name, r, m, p, rec, sec in rows:
        p_s = "  -  " if p != p else f"{p:.3f}"     # p != p is True for NaN
        rec_s = "  -  " if rec != rec else f"{rec:.3f}"
        print(f"{name:22s} {r:7.4f} {m:7.4f} {p_s:>7s} {rec_s:>7s} {sec:9.1f}")
    print("=" * 78)
    print("Lower RMSE/MAE is better. Higher P@10/R@10 is better.")


if __name__ == "__main__":
    main()
