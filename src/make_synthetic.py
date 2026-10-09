"""Generate a small MovieLens-schema dataset for smoke-testing the code (NOT for reported results)."""
import os, numpy as np, pandas as pd
from common import RAW

def main(n_users=3000, n_items=800, n_ratings=120000, seed=7):
    rng = np.random.default_rng(seed)
    os.makedirs(RAW, exist_ok=True)
    k = 6
    U = rng.normal(size=(n_users, k)); V = rng.normal(size=(n_items, k))
    pop = rng.zipf(1.6, n_items).astype(float); pop /= pop.sum()
    u = rng.integers(1, n_users + 1, n_ratings)
    i = rng.choice(np.arange(1, n_items + 1), n_ratings, p=pop)
    score = (U[u - 1] * V[i - 1]).sum(1) + rng.normal(0, 1, n_ratings)
    r = np.clip(np.round((3.4 + 0.7 * score) * 2) / 2, 0.5, 5.0)
    t0, t1 = 1_100_000_000, 1_560_000_000
    quit_t = rng.uniform(t0, t1 + 3e7, n_users)
    ts = rng.uniform(t0, t1, n_ratings).astype(np.int64)
    keep = ts < quit_t[u - 1]
    df = pd.DataFrame({"userId": u, "movieId": i, "rating": r, "timestamp": ts})[keep].drop_duplicates(["userId", "movieId"])
    df.to_csv(os.path.join(RAW, "ratings.csv"), index=False)
    genres = ["Action", "Comedy", "Drama", "Romance", "Sci-Fi", "Thriller", "Documentary", "Animation"]
    mv = pd.DataFrame({"movieId": np.arange(1, n_items + 1)})
    mv["title"] = ["Movie %d (%d)" % (m, 1980 + m % 40) for m in mv.movieId]
    mv["genres"] = ["|".join(rng.choice(genres, rng.integers(1, 4), replace=False)) for _ in range(n_items)]
    mv.to_csv(os.path.join(RAW, "movies.csv"), index=False)
    print("synthetic ratings:", len(df))

if __name__ == "__main__":
    main()
