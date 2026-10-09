#!/usr/bin/env bash
# Usage: ./run_all.sh [25m|small|synthetic]
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-25m}"
if [ "$MODE" = "synthetic" ]; then python src/make_synthetic.py; else python src/download_data.py "$MODE"; fi
python src/ingest.py
python src/train_als.py
python src/churn.py
python src/streaming_demo.py
python src/make_paper_assets.py
echo "Done. Build the paper: cd paper && pdflatex main && pdflatex main"
