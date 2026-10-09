# Scalable OTT Recommendation & Churn-Signal Pipeline (Apache Spark)

BDA Assignment 10 — Vidyalankar Institute of Technology, Dept. of Computer Engineering
**Authors:** [Author 1], [Author 2]  ·  **Paper:** `paper/main.tex` (IEEE conference format)

An end-to-end Big Data pipeline for an OTT use case, built on **PySpark** with the public
**MovieLens 25M** dataset as a proxy for streaming-platform feedback.

| Stage | Script | What it does |
|---|---|---|
| 0 | `src/download_data.py` | Downloads MovieLens (`25m` or `small`) |
| 1 | `src/ingest.py` | Explicit-schema CSV → de-duplicated Parquet |
| 2 | `src/features.py` | Min-count filtering + chronological 80/10/10 split |
| 3 | `src/train_als.py` | Baselines, ALS grid search, RMSE, Precision/Recall@10, catalogue coverage |
| 4 | `src/churn.py` | Engagement-decay churn signal (LogReg + GBT), pre-cutoff features only |
| 5 | `src/streaming_demo.py` | Structured Streaming demo serving top-10 lists |
| 6 | `src/make_paper_assets.py` | Writes `paper/results_macros.tex` and the validation figure from `results/*.json` |

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # needs Java 11/17/21 on PATH
./run_all.sh small        # ~1 min smoke run on ml-latest-small
./run_all.sh 25m          # full run for the paper numbers (~1 GB download)
```
On Windows without bash, run the scripts in `src/` in the order above (`cd src` first).

Useful environment variables: `SPARK_DRIVER_MEMORY` (default 4g; use 8g+ for 25M), `ALS_RANKS`
(default `10,20,50`), `ALS_REGS` (`0.05,0.1,0.2`), `ALS_ITER` (10), `EVAL_USERS` (5000), `SHUFFLE_PARTITIONS` (64).

`./run_all.sh synthetic` generates a tiny fake MovieLens-schema dataset to check the code runs.
**Never report synthetic numbers.**

## Building the paper
```bash
cd paper && pdflatex main && pdflatex main
```
(or upload the `paper/` folder to Overleaf). Until `run_all.sh` has been run, result cells show “–”.
After a real run, `results_macros.tex` and `fig_rank.pdf` are regenerated automatically.
Then re-read the `% EDIT` comment in `main.tex` (ranking discussion) and make the sentence match your numbers.

## Running on Google Colab (if your PC blocks Spark/pandas)
Clone the repo in a Colab notebook, `pip install pyspark==3.5.3 requests`, then run the scripts in `src/` in order.
To redo only the final ALS fit + evaluation after a finished grid search: `SKIP_GRID=1 python train_als.py`
(reuses the grid stored in `results/metrics_als.json`).

## Evaluation protocol
- **Chronological split** on global timestamps (no future leakage).
- **Cold start:** unseen users/items in test are dropped by ALS; the scored share is reported.
- **Ranking:** relevant = rating ≥ 4; evaluated on warm users and already-trained items only; already-seen titles removed; popularity baseline included.
- **RMSE:** baselines are also reported on exactly the rows ALS can score (fair comparison).
- **Churn proxy:** no events after the 80th-percentile time; features use only earlier events.
  This is an engagement proxy, not billing churn.

## Layout
```
src/        pipeline code           paper/   IEEE LaTeX source (+ main.pdf preview)
data/       raw/parquet (git-ignored)   results/ metrics_*.json written by the pipeline
```

## Contribution statement
- **[Author 1]:** ingestion, cleaning, split, churn features, repo/README
- **[Author 2]:** ALS modelling, tuning, evaluation, streaming demo, figures
- **Both:** architecture, paper writing and review

## References
Dataset: F. M. Harper and J. A. Konstan, "The MovieLens Datasets: History and Context," ACM TiiS, 2015.
