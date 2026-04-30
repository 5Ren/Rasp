import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/Users/ren/PycharmProjects/Rasp/08_Fuji/04_svm_test/03_results")

targets = [
    "Degree_of_roasting","Acid_taste","Bitter_taste","Astrigency_mouthfeel",
    "Cup_strength","Body","Aromatic_impact","Floral","Earthy","Tarry","Nutty","Pruney"
]

for tgt in targets:
    pi_path = BASE / f"pi_{tgt}.csv"
    df = pd.read_csv(pi_path)
    # importanceの大きい順に上位10件
    topk = df.sort_values("importance_mean", ascending=False).head(10)

    plt.figure()
    plt.barh(topk["feature"], topk["importance_mean"])
    plt.gca().invert_yaxis()  # 上位が上に来るように
    plt.title(f"Permutation importance (top 10): {tgt}")
    plt.xlabel("importance_mean")
    plt.tight_layout()
    plt.show()
