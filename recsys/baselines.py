"""Baselines: global mean and a user/item bias model."""

import numpy as np


class GlobalMean:
    def fit(self, train, n_users, n_items):
        _, _, r = train
        self.mu = float(r.mean())
        self.n_items = n_items
        return self

    def predict(self, users, items):
        return np.full(len(users), self.mu)

    def predict_all_items(self, user):
        return np.full(self.n_items, self.mu)


class BiasModel:
    """mu + b_u + b_i, learned by simple SGD."""

    def __init__(self, lr=0.005, reg=0.02, epochs=20, seed=0):
        self.lr, self.reg, self.epochs, self.seed = lr, reg, epochs, seed

    def fit(self, train, n_users, n_items):
        u, i, r = train
        self.mu = float(r.mean())
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)
        self.n_items = n_items
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(r))
        for _ in range(self.epochs):
            rng.shuffle(idx)
            for n in idx:
                uu, ii = u[n], i[n]
                err = r[n] - (self.mu + self.b_u[uu] + self.b_i[ii])
                self.b_u[uu] += self.lr * (err - self.reg * self.b_u[uu])
                self.b_i[ii] += self.lr * (err - self.reg * self.b_i[ii])
        return self

    def predict(self, users, items):
        return self.mu + self.b_u[users] + self.b_i[items]

    def predict_all_items(self, user):
        return self.mu + self.b_u[user] + self.b_i
