"""Turn results/*.json into LaTeX macros + a figure used by paper/main.tex.
Usage: python src/make_paper_assets.py [--placeholder]"""
import os, sys, json
from common import RESULTS, ROOT

PAPER = os.path.join(ROOT, "paper")
NAMES = ["dsRatings","dsUsers","dsItems","ingestSec","nTrain","nVal","nTest","rmseGlobal","rmseItem","rmseAls",
         "bestRank","bestReg","alsFitSec","alsCoverage","evalUsers","precPop","precAls","recPop","recAls",
         "covPop","covAls","churnUsers","churnRate","aucLR","aucGBT","prLR","prGBT","topFeatA","topFeatB","topFeatC","streamRows"]

def fmt(x, d=4): return f"{x:.{d}f}"
def comma(n): return f"{int(n):,}".replace(",", "{,}")
def tex(s): return str(s).replace("_", "\\_")

def build():
    L = lambda f: json.load(open(os.path.join(RESULTS, f)))
    ds, a, c = L("dataset_stats.json"), L("metrics_als.json"), L("metrics_churn.json")
    st = L("metrics_stream.json") if os.path.exists(os.path.join(RESULTS, "metrics_stream.json")) else {"rows": 0}
    tk, f = a["topk"], c["gbt_top_features"]
    return {
        "dsRatings": comma(ds["ratings"]), "dsUsers": comma(ds["users"]), "dsItems": comma(ds["items"]),
        "ingestSec": ds["ingest_seconds"], "nTrain": comma(a["split_counts"]["train"]),
        "nVal": comma(a["split_counts"]["val"]), "nTest": comma(a["split_counts"]["test"]),
        "rmseGlobal": fmt(a["baseline_global_mean"]["rmse"]), "rmseItem": fmt(a["baseline_item_mean"]["rmse"]),
        "rmseAls": fmt(a["als_test"]["rmse"]), "bestRank": a["als_best_params"]["rank"],
        "bestReg": a["als_best_params"]["regParam"], "alsFitSec": a["als_test"]["fit_seconds"],
        "alsCoverage": fmt(100 * a["als_test"]["coverage"], 1), "evalUsers": comma(tk["eval_users"]),
        "precPop": fmt(tk["popularity"]["precision"]), "precAls": fmt(tk["als"]["precision"]),
        "recPop": fmt(tk["popularity"]["recall"]), "recAls": fmt(tk["als"]["recall"]),
        "covPop": fmt(100 * tk["popularity"]["catalog_coverage"], 2), "covAls": fmt(100 * tk["als"]["catalog_coverage"], 2),
        "churnUsers": comma(c["users"]), "churnRate": fmt(100 * c["churn_rate"], 1),
        "aucLR": fmt(c["logistic_regression"]["auc_roc"]), "aucGBT": fmt(c["gbt"]["auc_roc"]),
        "prLR": fmt(c["logistic_regression"]["auc_pr"]), "prGBT": fmt(c["gbt"]["auc_pr"]),
        "topFeatA": tex(f[0][0]), "topFeatB": tex(f[1][0]), "topFeatC": tex(f[2][0]),
        "streamRows": comma(st.get("rows", 0)),
    }, a

def figure(a):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = a["als_grid"]; regs = sorted({x["regParam"] for x in g})
    plt.figure(figsize=(3.4, 2.4))
    for r in regs:
        pts = sorted([(x["rank"], x["val_rmse"]) for x in g if x["regParam"] == r])
        plt.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"$\\lambda$={r}")
    plt.xlabel("Latent rank"); plt.ylabel("Validation RMSE"); plt.legend(fontsize=7); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(PAPER, "fig_rank.pdf"))

def main():
    if "--placeholder" in sys.argv:
        vals = {k: "\\textbf{--}" for k in NAMES}
    else:
        vals, a = build(); figure(a)
    with open(os.path.join(PAPER, "results_macros.tex"), "w") as fh:
        for k in NAMES:
            fh.write(f"\\newcommand{{\\{k}}}{{{vals[k]}}}\n")
    print("wrote", os.path.join(PAPER, "results_macros.tex"))

if __name__ == "__main__":
    main()
