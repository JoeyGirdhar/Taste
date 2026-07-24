"""
Entry point. Runs the full model comparison on MovieLens 100K.

    python main.py

Other demos:
    python examples/similar_movies.py "Star Wars"   # "if you liked X, try Y"
    python examples/recommend_for_user.py 42         # personalized top-10
    python tests/test_recsys.py                      # fast sanity tests (no download)
"""

from examples.evaluate_all import main

if __name__ == "__main__":
    main()
