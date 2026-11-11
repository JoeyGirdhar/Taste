"""MovieLens 100K loading and train/test split. Downloads and caches the data on first use."""

import io
import os
import zipfile
import urllib.request

import numpy as np

ML_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def _cache_dir(cache_dir):
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "data", "ml-100k")
    return os.path.abspath(cache_dir)


def download_movielens(cache_dir=None):
    """Fetch and extract u.data (ratings) and u.item (titles). Cached after first run."""
    cache_dir = _cache_dir(cache_dir)
    if os.path.exists(os.path.join(cache_dir, "u.data")):
        return cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    print("Downloading MovieLens 100K (~5 MB, one time)...")
    raw = urllib.request.urlopen(ML_100K_URL).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if base in ("u.data", "u.item"):
                with z.open(name) as src, open(os.path.join(cache_dir, base), "wb") as dst:
                    dst.write(src.read())
    return cache_dir


def load_ratings(cache_dir=None):
    """Return (users, items, ratings) as 0-indexed int/float arrays."""
    cache_dir = download_movielens(cache_dir)
    data = np.loadtxt(os.path.join(cache_dir, "u.data"), dtype=int)
    users = data[:, 0] - 1
    items = data[:, 1] - 1
    ratings = data[:, 2].astype(float)
    return users, items, ratings


def load_titles(cache_dir=None):
    """Return {item_id (0-based): "Movie Title (Year)"}. u.item is latin-1 encoded."""
    cache_dir = download_movielens(cache_dir)
    titles = {}
    with open(os.path.join(cache_dir, "u.item"), encoding="latin-1") as f:
        for line in f:
            parts = line.split("|")
            titles[int(parts[0]) - 1] = parts[1]
    return titles


def train_test_split(users, items, ratings, test_frac=0.2, seed=0):
    """Seeded random split into (train_triples, test_triples)."""
    n = len(ratings)
    idx = np.random.default_rng(seed).permutation(n)
    cut = int(n * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]
    train = (users[tr], items[tr], ratings[tr])
    test = (users[te], items[te], ratings[te])
    return train, test


class MovieLens:
    """Convenience bundle: loads, splits, and exposes shape + titles."""

    def __init__(self, test_frac=0.2, seed=0, cache_dir=None):
        users, items, ratings = load_ratings(cache_dir)
        self.n_users = int(users.max()) + 1
        self.n_items = int(items.max()) + 1
        self.train, self.test = train_test_split(users, items, ratings, test_frac, seed)
        self.titles = load_titles(cache_dir)

    def title(self, item_id):
        return self.titles.get(item_id, f"<item {item_id}>")
