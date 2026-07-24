"""
Item-item collaborative filtering -- the classic "because you watched X" engine
(it's essentially what Amazon's "customers who bought this also bought" made
famous).

The idea, in one sentence: two movies are similar if the same people rate them
the same way. So we:

  1. Build an item x user rating matrix from the training data.
  2. Mean-center each user's ratings (subtract their average) so a generous
     rater and a harsh rater are comparable.
  3. Compute cosine similarity between every pair of items.
  4. To predict how user u would rate item i, look at the items u HAS rated,
     weight them by how similar they are to i, and take a weighted average.

The same similarity matrix also powers the fun demo: "movies most similar to
Toy Story" is just the top row of the similarity matrix.
"""

import numpy as np


class ItemCF:
    def __init__(self, k_neighbors=40):
        self.k = k_neighbors

    def fit(self, train, n_users, n_items):
        u, i, r = train
        self.n_items = n_items

        # Dense item x user matrix (1682 x 943 for ML-100K -> ~12 MB, fine).
        R = np.zeros((n_items, n_users))
        R[i, u] = r
        rated = R != 0

        # Mean-center by USER (column): removes "this person rates high/low" bias.
        user_sum = R.sum(axis=0)
        user_cnt = rated.sum(axis=0)
        user_mean = np.divide(user_sum, np.maximum(user_cnt, 1))
        Rc = np.where(rated, R - user_mean, 0.0)

        # Item mean rating (used as the prediction baseline before neighbor tweaks).
        item_sum = R.sum(axis=1)
        item_cnt = rated.sum(axis=1)
        self.item_mean = np.divide(item_sum, np.maximum(item_cnt, 1))
        self.global_mean = float(r.mean())

        # Cosine similarity between item rows of the centered matrix.
        norms = np.linalg.norm(Rc, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        normalized = Rc / norms
        self.sim = normalized @ normalized.T          # (n_items x n_items)
        np.fill_diagonal(self.sim, 0.0)               # an item isn't its own neighbor

        # Per-user list of (item, centered_rating) for fast prediction.
        self._user_rated = {}
        for uu, ii, rr in zip(u, i, r):
            self._user_rated.setdefault(uu, []).append((ii, rr - user_mean[uu]))
        self.user_mean = user_mean
        return self

    def _predict_one(self, user, item):
        # Adjusted-cosine prediction: baseline is the USER's mean (consistent with
        # how similarities were centered), plus a similarity-weighted average of
        # how this user deviated on neighboring items. Using the user mean as the
        # baseline -- rather than the item mean -- keeps obscure, thinly-rated
        # movies with inflated averages from dominating the recommendations.
        rated = self._user_rated.get(user)
        if not rated:
            return self.global_mean
        items_j = np.array([j for j, _ in rated])
        resid_j = np.array([c for _, c in rated])     # already r - user_mean[user]
        sims = self.sim[item, items_j]

        if self.k < len(sims):                        # keep only top-k neighbors
            top = np.argpartition(-np.abs(sims), self.k)[:self.k]
            sims, resid_j = sims[top], resid_j[top]

        denom = np.abs(sims).sum()
        if denom < 1e-8:
            return self.user_mean[user]
        pred = self.user_mean[user] + float((sims * resid_j).sum() / denom)
        return min(5.0, max(1.0, pred))               # ratings live in [1, 5]

    def predict(self, users, items):
        return np.array([self._predict_one(u, i) for u, i in zip(users, items)])

    def predict_all_items(self, user):
        return np.array([self._predict_one(user, i) for i in range(self.n_items)])

    def similar_items(self, item, n=10):
        """Top-n most similar items to `item`, as (item_id, similarity) pairs."""
        order = np.argsort(-self.sim[item])[:n]
        return [(int(j), float(self.sim[item, j])) for j in order]
