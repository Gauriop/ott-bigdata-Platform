"""Stage 4: engagement-decay churn signal.
Cutoff = 80th percentile of time. Features use ONLY events before the cutoff.
Label churn=1 if the user has no rating events after the cutoff (proxy for disengagement)."""
import os, json
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from common import get_spark, RESULTS, PARQUET

DAY = 86400

def main():
    spark = get_spark("churn")
    r = spark.read.parquet(os.path.join(PARQUET, "ratings"))
    movies = spark.read.parquet(os.path.join(PARQUET, "movies"))
    cut = r.approxQuantile("timestamp", [0.8], 0.001)[0]
    before = r.filter(F.col("timestamp") <= cut)
    after_users = r.filter(F.col("timestamp") > cut).select("userId").distinct().withColumn("active_after", F.lit(1))

    gen = (before.join(movies, "movieId").select("userId", F.explode(F.split("genres", r"\|")).alias("g"))
           .groupBy("userId").agg(F.countDistinct("g").alias("n_genres")))
    feats = before.groupBy("userId").agg(
        F.count("*").alias("n_ratings"),
        F.avg("rating").alias("avg_rating"),
        F.coalesce(F.stddev("rating"), F.lit(0.0)).alias("std_rating"),
        F.countDistinct("movieId").alias("n_titles"),
        ((F.max("timestamp") - F.min("timestamp")) / DAY).alias("span_days"),
        ((F.lit(cut) - F.max("timestamp")) / DAY).alias("days_since_last"),
        F.sum(F.when(F.col("timestamp") > cut - 30 * DAY, 1).otherwise(0)).alias("r_last30"),
        F.sum(F.when(F.col("timestamp") > cut - 90 * DAY, 1).otherwise(0)).alias("r_last90"),
    ).join(gen, "userId", "left").fillna(0)
    feats = feats.withColumn("recent_ratio", F.col("r_last90") / F.col("n_ratings"))
    df = (feats.filter("n_ratings >= 5").join(after_users, "userId", "left")
          .withColumn("label", F.when(F.col("active_after").isNull(), 1.0).otherwise(0.0))
          .drop("active_after")).cache()

    cols = ["n_ratings", "avg_rating", "std_rating", "n_titles", "span_days", "days_since_last",
            "r_last30", "r_last90", "n_genres", "recent_ratio"]
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    asm = VectorAssembler(inputCols=cols, outputCol="raw")
    sc = StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True)
    res = {"users": df.count(), "churn_rate": df.agg(F.avg("label")).first()[0], "features": cols}
    for name, clf in [("logistic_regression", LogisticRegression(maxIter=100)),
                      ("gbt", GBTClassifier(maxIter=40, maxDepth=5, seed=42))]:
        m = Pipeline(stages=[asm, sc, clf]).fit(train)
        p = m.transform(test)
        res[name] = {
            "auc_roc": BinaryClassificationEvaluator(metricName="areaUnderROC").evaluate(p),
            "auc_pr": BinaryClassificationEvaluator(metricName="areaUnderPR").evaluate(p)}
        if name == "gbt":
            imp = sorted(zip(cols, m.stages[-1].featureImportances.toArray().tolist()), key=lambda x: -x[1])
            res["gbt_top_features"] = imp[:5]
    json.dump(res, open(os.path.join(RESULTS, "metrics_churn.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))
    spark.stop()

if __name__ == "__main__":
    main()
