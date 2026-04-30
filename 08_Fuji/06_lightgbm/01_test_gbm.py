# -*- coding: utf-8 -*-
"""
LightGBM + GroupKFold + SHAP
コーヒー官能評価(12項目)を予測するための1ファイル版
"""

import os
from pathlib import Path
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from scipy.integrate import trapezoid
from scipy.stats import linregress

import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
# plt.use("Agg")  # 画面なし環境でも保存できる


# =========================================================
# 0. パス設定
# =========================================================
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

# データ一式の場所（いままで通り）
DATA_DIR = ROOT.parent / "02_Data_all"

PATH_Y   = DATA_DIR / "01_appraisers_evaluation.csv"
PATH_ETC = DATA_DIR / "02_etc_data.csv"
DIR_ORP  = DATA_DIR / "03_redox_potential"
DIR_CMOS = DATA_DIR / "04_cmos"
DIR_QCM  = DATA_DIR / "05_qcm"

TIMES = [0, 2, 4, 6]
TARGET_COLS = [
    "Degree_of_roasting","Acid_taste","Bitter_taste","Astrigency_mouthfeel",
    "Cup_strength","Body","Aromatic_impact","Floral","Earthy","Tarry","Nutty","Pruney"
]


# =========================================================
# 1. 時系列→特徴量 (30s以降)
# =========================================================
def extract_ts_features(df: pd.DataFrame, time_col="Time[s]"):
    """CMOS/QCMの1列を5つの統計量に変換する"""
    df = df[df[time_col] >= 30].copy()
    t = df[time_col].to_numpy()
    v = df.iloc[:, 1].to_numpy()

    if len(t) == 0:
        return {
            "baseline_30s": np.nan,
            "max_30s": np.nan,
            "delta_30s": np.nan,
            "auc_30s": np.nan,
            "slope_30s": np.nan,
        }

    # baseline: 30〜35s
    m = (t >= 30) & (t <= 35)
    if m.any():
        baseline = float(v[m].mean())
    else:
        baseline = float(v[0])

    vmax = float(v.max())
    delta = float(vmax - baseline)
    auc = float(trapezoid(v, t))

    # slope: 30〜40sで直線近似
    m2 = (t >= 30) & (t <= 40)
    if m2.sum() >= 2:
        slope, _, _, _, _ = linregress(t[m2], v[m2])
        slope = float(slope)
    else:
        slope = np.nan

    return {
        "baseline_30s": baseline,
        "max_30s": vmax,
        "delta_30s": delta,
        "auc_30s": auc,
        "slope_30s": slope,
    }


# =========================================================
# 2. 各データ読込
# =========================================================
def load_etc():
    df = pd.read_csv(PATH_ETC)
    df = df.set_index("Time[h]").loc[TIMES]
    return df


def _orp_to_bins(classes, amp, n_bins=16, prefix="orp"):
    cmin, cmax = classes.min(), classes.max()
    bins = np.linspace(cmin, cmax, n_bins + 1)
    inds = np.digitize(classes, bins) - 1
    feat = {}
    for b in range(n_bins):
        sel = amp[inds == b]
        feat[f"{prefix}_bin_{b:02d}"] = float(sel.mean()) if sel.size > 0 else 0.0
    feat[f"{prefix}_sum"] = float(amp.sum())
    feat[f"{prefix}_mean"] = float(amp.mean())
    feat[f"{prefix}_std"] = float(amp.std())
    feat[f"{prefix}_max"] = float(amp.max())
    feat[f"{prefix}_class_at_max"] = float(classes[amp.argmax()])
    return feat


def load_orp_for_time(t: int):
    f = DIR_ORP / f"sen_{t}.csv"
    df = pd.read_csv(f)
    classes = df["Class"].to_numpy()
    cols = [c for c in df.columns if c != "Class"]

    rows = []
    # mean
    amps = df[cols].to_numpy(dtype=float)
    amp_mean = amps.mean(axis=1)
    rows.append(("mean", _orp_to_bins(classes, amp_mean, prefix="orp")))
    # each N
    for c in cols:
        amp = df[c].to_numpy(dtype=float)
        rows.append((c, _orp_to_bins(classes, amp, prefix="orp")))
    return rows


def load_cmos_for_time(t: int):
    pattern = str(DIR_CMOS / f"{t}h_N*.csv")
    files = sorted(glob.glob(pattern))
    rows = []
    for fp in files:
        df = pd.read_csv(fp)
        feats_one = {}
        for m in range(1, 6):
            col = f"Membrane{m}"
            sub = df[["Time[s]", col]].copy()
            sub.columns = ["Time[s]", "Y"]
            fts = extract_ts_features(sub)
            for k, v in fts.items():
                feats_one[f"cmos_m{m}_{k}"] = v
        rows.append((Path(fp).stem, feats_one))

    if not rows:
        return [("mean", {})]

    # 平均行
    all_keys = set().union(*[set(d.keys()) for _, d in rows])
    mean_feats = {}
    for k in all_keys:
        vals = [d.get(k, np.nan) for _, d in rows]
        mean_feats[k] = float(np.nanmean(vals))
    out = [("mean", mean_feats)]
    out.extend(rows)
    return out


def load_qcm_for_time(t: int):
    pattern = str(DIR_QCM / f"{t}h_N*")
    folders = sorted(glob.glob(pattern))
    rows = []
    for folder in folders:
        graphs = sorted(glob.glob(os.path.join(folder, "graph_*.csv")))
        feats_rep = {}
        for g in graphs:
            gname = Path(g).stem
            df = pd.read_csv(g)
            fts = extract_ts_features(df, time_col="Time[s]")
            for k, v in fts.items():
                feats_rep[f"qcm_{gname}_{k}"] = v
        rows.append((Path(folder).name, feats_rep))

    if not rows:
        return [("mean", {})]

    all_keys = set().union(*[set(d.keys()) for _, d in rows])
    mean_feats = {}
    for k in all_keys:
        vals = [d.get(k, np.nan) for _, d in rows]
        mean_feats[k] = float(np.nanmean(vals))
    out = [("mean", mean_feats)]
    out.extend(rows)
    return out


def load_targets():
    df = pd.read_csv(PATH_Y)
    df = df.set_index("Time[h]").loc[TIMES]
    return df[TARGET_COLS]


# =========================================================
# 3. 拡張データセットを組み立てる
# =========================================================
def build_dataset():
    y_all = load_targets()
    etc = load_etc()

    X_rows = []
    y_rows = []
    groups = []

    for t in TIMES:
        etc_row = etc.loc[t].to_dict()
        orp_list = load_orp_for_time(t)
        cmos_list = load_cmos_for_time(t)
        qcm_list = load_qcm_for_time(t)

        # base行
        base = {}
        base.update(etc_row)
        base.update(orp_list[0][1])
        base.update(cmos_list[0][1])
        base.update(qcm_list[0][1])

        X_rows.append(base)
        y_rows.append(y_all.loc[t].to_list())
        groups.append(t)

        # ORP差し替え
        for name, feats in orp_list[1:]:
            f = dict(base)
            f.update(feats)
            X_rows.append(f)
            y_rows.append(y_all.loc[t].to_list())
            groups.append(t)

        # CMOS差し替え
        for name, feats in cmos_list[1:]:
            f = dict(base)
            f.update(feats)
            X_rows.append(f)
            y_rows.append(y_all.loc[t].to_list())
            groups.append(t)

        # QCM差し替え
        for name, feats in qcm_list[1:]:
            f = dict(base)
            f.update(feats)
            X_rows.append(f)
            y_rows.append(y_all.loc[t].to_list())
            groups.append(t)

    X = pd.DataFrame(X_rows)
    y = pd.DataFrame(y_rows, columns=TARGET_COLS)
    groups = np.array(groups)
    return X, y, groups


# =========================================================
# 4. 特徴選択（ホワイトリスト方式）
# =========================================================
def select_features(X: pd.DataFrame, groups: np.ndarray) -> pd.DataFrame:
    keep_cols = []
    for c in X.columns:
        # QCM: graph_12系
        if c.startswith("qcm_graph_12_"):
            keep_cols.append(c)
        # QCM: slope系
        elif c.startswith("qcm_graph_14_slope_30s"):
            keep_cols.append(c)
        elif c.startswith("qcm_graph_03_slope_30s"):
            keep_cols.append(c)
        elif c.startswith("qcm_graph_01_slope_30s"):
            keep_cols.append(c)
        elif c.startswith("qcm_graph_11_slope_30s"):
            keep_cols.append(c)
        elif c.startswith("qcm_graph_07_slope_30s"):
            keep_cols.append(c)
        # CMOS: m4系
        elif c.startswith("cmos_m4_"):
            keep_cols.append(c)
        # 液性
        elif c in ["pH", "Conductivity", "Brix_1", "Brix_n2"]:
            keep_cols.append(c)
        # ORP高域
        elif c in ["orp_bin_06", "orp_bin_07", "orp_max", "orp_class_at_max"]:
            keep_cols.append(c)

    X_sel = X[keep_cols].copy()
    # 時間フラグ
    X_sel["is_time0"] = (groups == 0).astype(int)
    return X_sel


# =========================================================
# 5. LightGBM + SHAP 本体
# =========================================================
# ・・・前半のデータ読み込み・build_datasetまでは同じとする・・・

def run_lgbm_with_shap():
    X, y, groups = build_dataset()

    # 数値だけにして欠損を埋める
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.mean(numeric_only=True))

    # ここで特徴量を追加するのはOK
    X["group_id"] = groups
    X["is_time0"] = (groups == 0).astype(int)

    # ★ここで「使う特徴量の順番」を固定しておく
    feature_cols = X.columns.tolist()

    gkf = GroupKFold(n_splits=4)

    for tgt in TARGET_COLS:
        print(f"\n===== Target: {tgt} =====")
        y_t = y[tgt].to_numpy()

        fold_maes = []
        fold_models = []
        fold_valid_idx = []

        for tr_idx, te_idx in gkf.split(X, y_t, groups):
            # ★常に同じ順番・同じ列で切る
            X_tr = X.iloc[tr_idx][feature_cols]
            X_te = X.iloc[te_idx][feature_cols]
            y_tr, y_te = y_t[tr_idx], y_t[te_idx]

            model = lgb.LGBMRegressor(
                objective="regression",
                learning_rate=0.05,
                n_estimators=500,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                min_child_samples=1,
                min_data_in_bin=1,
                max_bin=255,
            )
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_te, y_te)],
                eval_metric="l1",
            )

            y_pred = model.predict(X_te)
            mae = mean_absolute_error(y_te, y_pred)
            fold_maes.append(mae)
            fold_models.append(model)
            fold_valid_idx.append(te_idx)

        print(f"CV MAE: {np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f}")

        # ここから可視化パート
        model0 = fold_models[0]
        valid_idx0 = fold_valid_idx[0]
        # ★ここでも必ず同じfeature_colsでそろえる
        X_valid0 = X.iloc[valid_idx0][feature_cols]

        # SHAP
        explainer = shap.TreeExplainer(model0)
        shap_vals = explainer.shap_values(X_valid0)

        # SHAPをCSVに
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False)
        shap_df.to_csv(ROOT / f"shap_{tgt}.csv", index=False)

        # SHAP summary plot
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_vals, X_valid0, show=False, plot_size=None)
        plt.tight_layout()
        plt.savefig(ROOT / f"shap_{tgt}.png", dpi=200)
        plt.close()

        # Feature importance
        fi = pd.DataFrame({
            "feature": feature_cols,
            "importance": model0.booster_.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False)
        fi.to_csv(ROOT / f"featimp_{tgt}.csv", index=False)

        top_n = 20 if len(fi) > 20 else len(fi)
        plt.figure(figsize=(8, 6))
        plt.barh(fi["feature"].iloc[:top_n][::-1], fi["importance"].iloc[:top_n][::-1])
        plt.xlabel("gain")
        plt.title(f"Feature importance: {tgt}")
        plt.tight_layout()
        plt.savefig(ROOT / f"featimp_{tgt}.png", dpi=200)
        plt.close()

    print("\nAll targets finished.")



# =========================================================
# main
# =========================================================
if __name__ == "__main__":
    run_lgbm_with_shap()
