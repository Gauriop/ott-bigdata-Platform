"""Shared helpers: Spark session and paths."""
import os
from pyspark.sql import SparkSession

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "raw")
PARQUET = os.path.join(ROOT, "data", "parquet")
RESULTS = os.path.join(ROOT, "results")


def get_spark(app="OTT-BigData", shuffle_partitions=None):
    mem = os.environ.get("SPARK_DRIVER_MEMORY", "4g")
    parts = str(shuffle_partitions or os.environ.get("SHUFFLE_PARTITIONS", "64"))
    spark = (
        SparkSession.builder.appName(app)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config("spark.driver.memory", mem)
        .config("spark.sql.shuffle.partitions", parts)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
