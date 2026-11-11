"""Rating metrics (RMSE, MAE) and ranking metrics (precision/recall at k)."""

import numpy as np


def rmse(pred, actual):
    """Root mean squared error -- penalizes big misses hard."""
    pred, actual = np.asarray(pred), np.asarray(actual)
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def mae(pred, actual):
    """Mean absolute error -- average miss in 'stars'."""
    pred, actual = np.asarray(pred), np.asarray(actual)
    return float(np.mean(np.abs(pred - actual)))


def precision_recall_at_k(model, train, test, n_items, k=10, relevant_threshold=4.0):
    """Average Precision@K and Recall@K over users who have >=1 'liked' test item.

    For each user we:
      1. hide the items they rated in train (can't recommend the already-seen),
      2. score every remaining item with the model,
      3. take the top K,
      4. count how many are 'relevant' (test rating >= threshold).

    Precision@K = relevant hits / K
    Recall@K    = relevant hits / total relevant items for that user
    """
    tr_u, tr_i, _ = train
    te_u, te_i, te_r = test

    # Items each user already rated in train (to exclude from recommendations).
    seen = {}
    for u, i in zip(tr_u, tr_i):
        seen.setdefault(u, set()).add(i)

    # Each user's relevant (highly rated) test items.
    relevant = {}
    for u, i, r in zip(te_u, te_i, te_r):
        if r >= relevant_threshold:
            relevant.setdefault(u, set()).add(i)

    precisions, recalls = [], []
    all_items = np.arange(n_items)
    for u, rel_items in relevant.items():
        scores = model.predict_all_items(u)
        scores = scores.copy()
        for i in seen.get(u, ()):
            scores[i] = -np.inf
        top_k = all_items[np.argsort(-scores)[:k]]
        hits = len(set(top_k.tolist()) & rel_items)
        precisions.append(hits / k)
        recalls.append(hits / len(rel_items))

    return float(np.mean(precisions)), float(np.mean(recalls))
