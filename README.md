# Movie Recommender From Scratch

Four recommendation algorithms - from a dumb baseline to Netflix-Prize-style
**matrix factorization** - built from scratch in NumPy and compared head-to-head
on the real **MovieLens 100K** dataset (943 users, 1,682 movies, 100,000 ratings).

No `surprise`, no `lightfm`, no black boxes. Collaborative filtering, matrix
factorization, and the evaluation harness are all implemented and measured here.

## The "if you liked X, try Y" demo

Item-item collaborative filtering, asked for movies similar to **Star Wars (1977)**:

```
Because you liked "Star Wars (1977)":
   0.545   Return of the Jedi (1983)
   0.531   Empire Strikes Back, The (1980)
   0.453   Raiders of the Lost Ark (1981)
   0.309   Princess Bride, The (1987)
   0.289   Usual Suspects, The (1995)
```

Nobody told the model these are related - it discovered the whole trilogy
(plus adjacent adventure films) purely from how people rated them together.

## Results - measured, not claimed

Same 80/20 train/test split for every model. Lower RMSE/MAE is better; higher
Precision@10 / Recall@10 is better.

| Model | RMSE | MAE | P@10 | R@10 |
|---|---|---|---|---|
| Global Mean (baseline) | 1.1296 | 0.9493 | - | - |
| Bias (μ + bᵤ + bᵢ) | 0.9432 | 0.7455 | 0.067 | 0.054 |
| Item-Item CF | 0.9398 | 0.7374 | 0.046 | 0.022 |
| **Matrix Factorization** | **0.9368** | 0.7399 | **0.075** | **0.059** |

**What the numbers say:** just predicting the global average is worth beating by
~17% RMSE. Matrix factorization - learning a hidden "taste vector" per user and
per movie - wins on both rating accuracy *and* ranking quality, which is why it
became the workhorse of modern recommenders.

## Quickstart

```bash
pip install -r requirements.txt
python main.py                                   # trains all 4 models, prints the table
python examples/similar_movies.py "Star Wars"    # "if you liked X, try Y"
python examples/recommend_for_user.py 42         # personalized top-10 for a user
python tests/test_recsys.py                      # fast sanity tests (no download)
```

Only dependency is NumPy. MovieLens 100K (~5 MB) downloads once and is cached.

## How each model works

- **Global Mean** - predicts the overall average for everyone. The yardstick.
- **Bias model** - `μ + bᵤ + bᵢ`, learned by SGD. Captures "Alice rates everything
  high" and "this movie is widely loved." Fast and surprisingly strong.
- **Item-item CF** - two movies are similar if the same people rate them alike.
  Uses **adjusted-cosine** similarity (ratings centered per user so a generous
  and a harsh rater are comparable), then predicts from a user's nearest-neighbor
  items. Also powers the "similar movies" demo.
- **Matrix factorization** - learns a length-`k` latent vector for every user and
  movie so that `μ + bᵤ + bᵢ + pᵤ · qᵢ` reproduces the known ratings. Trained with
  the same SGD + L2-regularization loop as a neural net. The factors self-organize
  into taste dimensions (action vs. romance, mainstream vs. arthouse) without ever
  being told what they mean.

## Project structure

```
recsys/
  data.py         MovieLens 100K loader (download + cache + split)
  metrics.py      RMSE, MAE, Precision@K, Recall@K
  baselines.py    GlobalMean, BiasModel
  item_cf.py      item-item collaborative filtering (adjusted cosine)
  mf.py           matrix factorization (SGD with biases + latent factors)
examples/
  evaluate_all.py      the head-to-head comparison table
  similar_movies.py    "if you liked X, try Y"
  recommend_for_user.py  personalized top-10 for one user
tests/
  test_recsys.py       metric checks + an MF-recovers-latent-factors test
main.py                runs evaluate_all
```

## A note on the ranking metrics

RMSE and ranking (P@K) don't always agree - a model can predict ratings well yet
order the top of the list poorly, because users only ever see the *top* of the
list. That's exactly why both are reported. It's also why the first version of the
item-CF here scored well on RMSE but poorly on P@10 until the prediction baseline
was fixed; the commit history shows that fix.

## Things to try (make it yours)

- Tune `n_factors`, `lr`, `reg`, `epochs` in `mf.py` and watch RMSE move.
- Add **user-user CF** and compare it to item-item.
- Swap the random split for a **time-based split** (train on older ratings, test
  on newer) - a more realistic and harder test.
- Add **NDCG@K** to `metrics.py` for a ranking metric that rewards putting the
  best items highest.
- Upgrade to **MovieLens 1M** by changing one URL and see the numbers improve.

## What this is - and isn't

- **Is:** a clear, correct, measured implementation of the core recommender
  algorithms, good for understanding how they actually work and differ.
- **Isn't:** a production system. It's in-memory, single-machine, and retrains
  from scratch. No cold-start handling, no online updates, no content features.

## Requirements

- Python 3.9+
- NumPy (only dependency)
