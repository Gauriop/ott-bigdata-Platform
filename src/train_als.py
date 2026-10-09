"""Stage 3: baselines + ALS (grid search on validation) + RMSE / Precision@K / Recall@K on a time-based test set."""
import os, json, time, itertools, sys
import numpy as np, pandas as pd
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from common import get_spark, RESULTS, PARQUET
from features import load_clean, time_split

K = 10
REL = 4.0          # rating >= 4 counts as "relevant"
EVAL_USERS = int(os.environ.get("EVAL_USERS", "5000"))
RANKS = [int(x) for x in os.environ.get("ALS_RANKS", "10,20,50").split(",")]
REGS = [float(x) for x in os.environ.get("ALS_REGS", "0.05,0.1,0.2").split(",")]
MAX_ITER = int(os.environ.get("ALS_ITER", "10"))


def rmse(df, col="prediction"):
    return RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol=col).evaluate(df)


def topk_metrics(recs, truth, k=K):
    """recs: {user: [items ranked]}, truth: {user: set(relevant items)}"""
    p, r, n = [], [], 0
    for u, rel in truth.items():
        if not rel:
            continue
        top = recs.get(u, [])[:k]
        hit = len(set(top) & rel)
        p.append(hit / k); r.append(hit / len(rel)); n += 1
    return {"precision": float(np.mean(p)), "recall": float(np.mean(r)), "users": n}


def main():
    spark = get_spark("als")
    os.makedirs(RESULTS, exist_ok=True)
    data = load_clean(spark).cache()
    train, val, test, cut = time_split(data)
    train.cache(); val.cache(); test.cache()
    out = {"split_counts": {"train": train.count(), "val": val.count(), "test": test.count()}}

    # ---------- baselines (RMSE) ----------
    gmean = train.agg(F.avg("rating")).first()[0]
    item_mean = train.groupBy("movieId").agg(F.avg("rating").alias("im"))
    base_g = test.withColumn("prediction", F.lit(float(gmean)))
    base_i = test.join(item_mean, "movieId", "left").withColumn(
        "prediction", F.coalesce("im", F.lit(float(gmean))))
    out["baseline_global_mean"] = {"rmse": rmse(base_g)}
    out["baseline_item_mean"] = {"rmse": rmse(base_i)}

    # ---------- ALS grid search on validation ----------
    prev = os.path.join(RESULTS, "metrics_als.json")
    if os.environ.get("SKIP_GRID") == "1" and os.path.exists(prev):
        grid = json.load(open(prev))["als_grid"]      # reuse a finished search
        print("Reusing existing grid from metrics_als.json", flush=True)
    else:
        grid = []
        for rank, reg in itertools.product(RANKS, REGS):
            t0 = time.time()
            als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", rank=rank, regParam=reg,
                      maxIter=MAX_ITER, coldStartStrategy="drop", nonnegative=True, seed=42)
            m = als.fit(train)
            v = rmse(m.transform(val))
            grid.append({"rank": rank, "regParam": reg, "val_rmse": v, "fit_seconds": round(time.time() - t0, 1)})
            print(grid[-1], flush=True)
    out["als_grid"] = grid
    best = min(grid, key=lambda g: g["val_rmse"])
    out["als_best_params"] = {"rank": best["rank"], "regParam": best["regParam"], "maxIter": MAX_ITER}

    # ---------- retrain on train+val, evaluate on test ----------
    t0 = time.time()
    tv = train.union(val)
    final = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", rank=best["rank"],
                regParam=best["regParam"], maxIter=MAX_ITER, coldStartStrategy="drop",
                nonnegative=True, seed=42).fit(tv)
    out["als_test"] = {"rmse": rmse(final.transform(test)), "fit_seconds": round(time.time() - t0, 1)}
    scored = final.transform(test).cache()
    cov = scored.count() / max(out["split_counts"]["test"], 1)
    out["als_test"]["coverage"] = cov   # share of test rows scored (cold-start rows are dropped)
    # fair comparison: baselines restricted to exactly the rows ALS scored
    keys = scored.select("userId", "movieId")
    out["baseline_global_mean_covered"] = {"rmse": rmse(base_g.join(keys, ["userId", "movieId"]))}
    out["baseline_item_mean_covered"] = {"rmse": rmse(base_i.join(keys, ["userId", "movieId"]))}

    # ---------- top-K ranking evaluation ----------
    # Evaluate on "warm" users (present in train+val) and items the model could ever recommend.
    # Users/items unseen in training cannot be scored by collaborative filtering at all.
    warm_users = tv.select("userId").distinct()
    warm_items = tv.select("movieId").distinct()
    rel_all = test.filter(F.col("rating") >= REL)
    rel = rel_all.join(warm_items, "movieId").join(warm_users, "userId")
    out["topk_pool"] = {"relevant_events_all": rel_all.count(), "relevant_events_warm": rel.count()}
    users = (rel.select("userId").distinct().orderBy(F.rand(1)).limit(EVAL_USERS)).cache()
    truth_pd = rel.join(users, "userId").select("userId", "movieId").toPandas()
    truth = truth_pd.groupby("userId")["movieId"].apply(set).to_dict()
    seen_pd = tv.join(users, "userId").select("userId", "movieId").toPandas()
    seen = seen_pd.groupby("userId")["movieId"].apply(set).to_dict()

    # ALS recs: take 200 candidates, drop already-seen, keep top-K
    recs_df = final.recommendForUserSubset(users, 200)
    rp = recs_df.select("userId", F.explode("recommendations").alias("r")).select(
        "userId", F.col("r.movieId").alias("movieId"), F.col("r.rating").alias("score")).toPandas()
    rp = rp.sort_values(["userId", "score"], ascending=[True, False])
    als_recs = {}
    for u, g in rp.groupby("userId"):
        s = seen.get(u, set())
        als_recs[u] = [m for m in g["movieId"] if m not in s][:K]

    # Popularity baseline
    pop = tv.groupBy("movieId").count().orderBy(F.desc("count")).limit(2000).toPandas()["movieId"].tolist()
    pop_recs = {u: [m for m in pop if m not in seen.get(u, set())][:K] for u in truth}

    out["topk"] = {
        "k": K, "relevance_threshold": REL, "eval_users": len(truth),
        "popularity": topk_metrics(pop_recs, truth),
        "als": topk_metrics(als_recs, truth),
    }
    # catalogue coverage of ALS recommendations
    n_items = data.select("movieId").distinct().count()
    out["topk"]["als"]["catalog_coverage"] = len({m for v in als_recs.values() for m in v}) / n_items
    out["topk"]["popularity"]["catalog_coverage"] = len({m for v in pop_recs.values() for m in v}) / n_items

    # persist a sample of top-10 recommendations for the streaming demo / README
    movies = spark.read.parquet(os.path.join(PARQUET, "movies"))
    final.recommendForAllUsers(K).write.mode("overwrite").parquet(os.path.join(RESULTS, "recs"))
    json.dump(out, open(os.path.join(RESULTS, "metrics_als.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))
    spark.stop()

if __name__ == "__main__":
    main()
