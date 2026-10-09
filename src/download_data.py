"""Download MovieLens into data/raw/.  Usage: python src/download_data.py [small|25m]"""
import os, sys, zipfile, requests
from common import RAW

URLS = {
    "small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
}

def main(kind="25m"):
    os.makedirs(RAW, exist_ok=True)
    zpath = os.path.join(RAW, f"ml-{kind}.zip")
    if not os.path.exists(zpath):
        print("Downloading", URLS[kind])
        with requests.get(URLS[kind], stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(zpath, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    with zipfile.ZipFile(zpath) as z:
        for m in z.namelist():
            name = os.path.basename(m)
            if name in ("ratings.csv", "movies.csv") and not os.path.exists(os.path.join(RAW, name)):
                with z.open(m) as src, open(os.path.join(RAW, name), "wb") as dst:
                    dst.write(src.read())
    print("Raw CSVs in", RAW)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "25m")
