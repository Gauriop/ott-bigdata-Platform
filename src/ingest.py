"""Stage 1: ingest raw CSVs with explicit schemas and persist as Parquet."""
import os, time, json
from pyspark.sql import functions as F
from pyspark.sql.types import *
from common import get_spark, RAW, PARQUET, RESULTS

RATINGS = StructType([
    StructField("userId", IntegerType()), StructField("movieId", IntegerType()),
    StructField("rating", FloatType()), StructField("timestamp", LongType())])
MOVIES = StructType([
    StructField("movieId", IntegerType()), StructField("title", StringType()),
    StructField("genres", StringType())])

def main():
    spark = get_spark("ingest")
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    ratings = spark.read.csv(os.path.join(RAW, "ratings.csv"), header=True, schema=RATINGS)
    movies = spark.read.csv(os.path.join(RAW, "movies.csv"), header=True, schema=MOVIES)
    ratings = ratings.dropna().dropDuplicates(["userId", "movieId"])
    ratings.write.mode("overwrite").parquet(os.path.join(PARQUET, "ratings"))
    movies.write.mode("overwrite").parquet(os.path.join(PARQUET, "movies"))
    r = spark.read.parquet(os.path.join(PARQUET, "ratings"))
    s = r.agg(F.count("*").alias("ratings"), F.countDistinct("userId").alias("users"),
              F.countDistinct("movieId").alias("items")).first()
    info = {"ratings": s["ratings"], "users": s["users"], "items": s["items"],
            "ingest_seconds": round(time.time() - t0, 1)}
    json.dump(info, open(os.path.join(RESULTS, "dataset_stats.json"), "w"), indent=2)
    print(info)
    spark.stop()

if __name__ == "__main__":
    main()
