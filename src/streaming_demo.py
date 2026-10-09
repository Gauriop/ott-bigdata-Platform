"""Stage 5 (optional): Structured Streaming demo.
A synthetic 'viewing events' stream (rate source) is joined with the precomputed ALS top-10 table
to serve recommendations per micro-batch. Demonstrates the velocity dimension; not used for reported metrics."""
import os, json, time
from pyspark.sql import functions as F
from common import get_spark, RESULTS

def main(seconds=15):
    spark = get_spark("stream-demo", shuffle_partitions=8)
    recs = spark.read.parquet(os.path.join(RESULTS, "recs")).cache()
    ids = [r[0] for r in recs.select("userId").limit(1000).collect()]
    lut = spark.createDataFrame([(i, u) for i, u in enumerate(ids)], ["idx", "userId"])
    events = (spark.readStream.format("rate").option("rowsPerSecond", 200).load()
              .withColumn("idx", (F.col("value") % len(ids)).cast("int")))
    served = events.join(lut, "idx").join(recs, "userId")
    stats = {"batches": 0, "rows": 0}

    def handle(batch_df, batch_id):
        n = batch_df.count()
        stats["batches"] += 1; stats["rows"] += n
        print(f"micro-batch {batch_id}: served {n} recommendation lists", flush=True)

    q = served.writeStream.foreachBatch(handle).trigger(processingTime="2 seconds").start()
    time.sleep(seconds); q.stop()
    json.dump(stats, open(os.path.join(RESULTS, "metrics_stream.json"), "w"), indent=2)
    print(stats)
    spark.stop()

if __name__ == "__main__":
    main()
