"""
The fun demo: "if you liked X, here are movies like it."

Trains item-item CF, then prints the nearest neighbors for a few well-known
films. Pass your own title (or part of one) as an argument.

Run:  python examples/similar_movies.py
      python examples/similar_movies.py "Star Wars"
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from recsys import MovieLens, ItemCF


def find_item(data, query):
    """First movie whose title contains `query` (case-insensitive)."""
    q = query.lower()
    for item_id, title in data.titles.items():
        if q in title.lower():
            return item_id
    return None


def show_similar(data, model, title_query):
    item_id = find_item(data, title_query)
    if item_id is None:
        print(f'No movie matching "{title_query}".')
        return
    print(f'\nBecause you liked "{data.title(item_id)}":')
    for j, sim in model.similar_items(item_id, n=8):
        print(f"   {sim:5.3f}   {data.title(j)}")


def main():
    data = MovieLens(seed=0)
    print("Training item-item collaborative filtering...")
    model = ItemCF(k_neighbors=40).fit(data.train, data.n_users, data.n_items)

    queries = sys.argv[1:] or ["Toy Story", "Star Wars", "Titanic", "GoldenEye"]
    for q in queries:
        show_similar(data, model, q)


if __name__ == "__main__":
    main()
