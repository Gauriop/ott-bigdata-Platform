"""Stage 2: cleaning, cold-start filtering and a leak-free time-based split."""
import os
from pyspark.sql import functions as F
from common import PARQUET

def load_clean(spark, min_user=5, min_item=5):
    r = spark.read.parquet(os.path.join(PARQUET, "ratings"))
    ucnt = r.groupBy("userId").count().filter(F.col("count") >= min_user).select("userId")
    icnt = r.groupBy("movieId").count().filter(F.col("count") >= min_item).select("movieId")
    return r.join(ucnt, "userId").join(icnt, "movieId")

def time_split(df, train=0.8, val=0.1):
    """Chronological split on global timestamp quantiles (no future leakage)."""
    q1, q2 = df.approxQuantile("timestamp", [train, train + val], 0.001)
    tr = df.filter(F.col("timestamp") <= q1)
    va = df.filter((F.col("timestamp") > q1) & (F.col("timestamp") <= q2))
    te = df.filter(F.col("timestamp") > q2)
    return tr, va, te, (q1, q2)
