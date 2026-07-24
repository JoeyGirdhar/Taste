"""
A movie recommender built from scratch on MovieLens 100K.

    from recsys import MovieLens
    from recsys import GlobalMean, BiasModel, ItemCF, MatrixFactorization
    from recsys.metrics import rmse, mae, precision_recall_at_k
"""

from .data import MovieLens
from .baselines import GlobalMean, BiasModel
from .item_cf import ItemCF
from .mf import MatrixFactorization

__all__ = [
    "MovieLens",
    "GlobalMean",
    "BiasModel",
    "ItemCF",
    "MatrixFactorization",
]
