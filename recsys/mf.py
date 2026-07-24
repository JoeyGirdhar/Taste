"""
Matrix factorization -- the approach that won the Netflix Prize era.

Instead of comparing items directly, we LEARN a short vector of hidden
"factors" for every user and every movie. You never tell the model what the
factors mean, but they end up encoding things like "how much action vs. romance"
or "mainstream vs. arthouse." A user's predicted rating for a movie is:

    r_hat(u, i) = mu + b_u + b_i + p_u · q_i

where p_u and q_i are the learned factor vectors and the dot product measures
how well the user's tastes line up with the movie's traits.

We train it with plain SGD: for each known rating, nudge every parameter a
little in the direction that reduces the squared error, with L2 regularization
so the factors don't blow up. This is the same gradient-descent loop as the
neural-net project, applied to a different model.
"""

import numpy as np

from .metrics import rmse


class MatrixFactorization:
    def __init__(self, n_factors=40, lr=0.005, reg=0.02, epochs=25, seed=0):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.seed = seed

    def fit(self, train, n_users, n_items, val=None, verbose=False):
        u, i, r = train
        rng = np.random.default_rng(self.seed)

        self.mu = float(r.mean())
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)
        # Small random init breaks symmetry so factors learn different things.
        self.P = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = rng.normal(0, 0.1, (n_items, self.n_factors))
        self.n_items = n_items

        idx = np.arange(len(r))
        for epoch in range(1, self.epochs + 1):
            rng.shuffle(idx)
            for n in idx:
                uu, ii = u[n], i[n]
                pred = self.mu + self.b_u[uu] + self.b_i[ii] + self.P[uu] @ self.Q[ii]
                err = r[n] - pred
                # Gradient step (with L2 reg) on every parameter touching this rating.
                self.b_u[uu] += self.lr * (err - self.reg * self.b_u[uu])
                self.b_i[ii] += self.lr * (err - self.reg * self.b_i[ii])
                p_old = self.P[uu].copy()
                self.P[uu] += self.lr * (err * self.Q[ii] - self.reg * self.P[uu])
                self.Q[ii] += self.lr * (err * p_old - self.reg * self.Q[ii])

            if verbose and val is not None:
                vp = self.predict(val[0], val[1])
                print(f"  epoch {epoch:2d}  val RMSE {rmse(vp, val[2]):.4f}")
        return self

    def predict(self, users, items):
        users, items = np.asarray(users), np.asarray(items)
        dot = np.sum(self.P[users] * self.Q[items], axis=1)
        return self.mu + self.b_u[users] + self.b_i[items] + dot

    def predict_all_items(self, user):
        return self.mu + self.b_u[user] + self.b_i + self.Q @ self.P[user]
