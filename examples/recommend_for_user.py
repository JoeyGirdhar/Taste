"""Top 10 recommendations for one user (matrix factorization).

Usage: python examples/recommend_for_user.py [user_id]"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from recsys import MovieLens, MatrixFactorization


def main():
    data = MovieLens(seed=0)
    print("Training matrix factorization...")
    model = MatrixFactorization(n_factors=40, epochs=25).fit(
        data.train, data.n_users, data.n_items
    )

    user = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    tr_u, tr_i, tr_r = data.train
    mask = tr_u == user
    seen_items = tr_i[mask]
    seen_ratings = tr_r[mask]

    print(f"\n=== User {user} ===")
    print("Top movies they rated highly (from training data):")
    for i in seen_items[np.argsort(-seen_ratings)][:5]:
        print(f"   {data.title(i)}")

    scores = model.predict_all_items(user)
    scores[seen_items] = -np.inf
    top = np.argsort(-scores)[:10]

    print("\nRecommended for them (predicted favorites they haven't seen):")
    for i in top:
        print(f"   {scores[i]:.2f}   {data.title(i)}")


if __name__ == "__main__":
    main()
