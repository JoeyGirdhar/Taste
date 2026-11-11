"""Item-item collaborative filtering with adjusted cosine similarity."""

import numpy as np


class ItemCF:
    def __init__(self, k_neighbors=40):
        self.k = k_neighbors

    def fit(self, train, n_users, n_items):
        u, i, r = train
        self.n_items = n_items

        # dense item x user rating matrix
        R = np.zeros((n_items, n_users))
        R[i, u] = r
        rated = R != 0

        # center each user's ratings
        user_sum = R.sum(axis=0)
        user_cnt = rated.sum(axis=0)
        user_mean = np.divide(user_sum, np.maximum(user_cnt, 1))
        Rc = np.where(rated, R - user_mean, 0.0)

        item_sum = R.sum(axis=1)
        item_cnt = rated.sum(axis=1)
        self.item_mean = np.divide(item_sum, np.maximum(item_cnt, 1))
        self.global_mean = float(r.mean())

        norms = np.linalg.norm(Rc, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        normalized = Rc / norms
        self.sim = normalized @ normalized.T
        np.fill_diagonal(self.sim, 0.0)

        self._user_rated = {}
        for uu, ii, rr in zip(u, i, r):
            self._user_rated.setdefault(uu, []).append((ii, rr - user_mean[uu]))
        self.user_mean = user_mean
        return self

    def _predict_one(self, user, item):
        # Use the user's mean as the baseline. The item mean let obscure,
        # thinly rated movies dominate the rankings.
        rated = self._user_rated.get(user)
        if not rated:
            return self.global_mean
        items_j = np.array([j for j, _ in rated])
        resid_j = np.array([c for _, c in rated])
        sims = self.sim[item, items_j]

        if self.k < len(sims):
            top = np.argpartition(-np.abs(sims), self.k)[:self.k]
            sims, resid_j = sims[top], resid_j[top]

        denom = np.abs(sims).sum()
        if denom < 1e-8:
            return self.user_mean[user]
        pred = self.user_mean[user] + float((sims * resid_j).sum() / denom)
        return min(5.0, max(1.0, pred))

    def predict(self, users, items):
        return np.array([self._predict_one(u, i) for u, i in zip(users, items)])

    def predict_all_items(self, user):
        return np.array([self._predict_one(user, i) for i in range(self.n_items)])

    def similar_items(self, item, n=10):
        """Top-n most similar items to `item`, as (item_id, similarity) pairs."""
        order = np.argsort(-self.sim[item])[:n]
        return [(int(j), float(self.sim[item, j])) for j in order]
