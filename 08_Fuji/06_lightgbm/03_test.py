# -*- coding: utf-8 -*-
"""
LightGBM + GroupKFold + SHAP
コーヒー官能評価(12項目)を予測するための1ファイル版
簡易ハイパーパラメータ探索つき（verbose対応版）
"""

import os
from pathlib import Path
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.integrate import trapezoid
from scipy.stats import linregress

import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

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


def extract_ts_features(df: pd.DataFrame, time_col="Time[s]"):
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

    m = (t >= 30) & (t <= 35)
    if m.any():
        baseline = float(v[m].mean())
    else:
        baseline = float(v[0])

    vmax = float(v.max())
    delta = float(vmax - baseline)
    auc = float(trapezoid(v, t))

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
    amps = df[cols].to_numpy(dtype=float)
    amp_mean = amps.mean(axis=1)
    rows.append(("mean", _orp_to_bins(classes, amp_mean, prefix="orp")))
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

        base = {}
        base.update(etc_row)
        base.update(orp_list[0][1])
        base.update(cmos_list[0][1])
        base.update(qcm_list[0][1])

        X_rows.append(base)
        y_rows.append(y_all.loc[t].to_list())
        groups.append(t)

        for name, feats in orp_list[1:]:
            f = dict(base)
            f.update(feats)
            X_rows.append(f)
            y_rows.append(y_all.loc[t].to_list())
            groups.append(t)

        for name, feats in cmos_list[1:]:
            f = dict(base)
            f.update(feats)
            X_rows.append(f)
            y_rows.append(y_all.loc[t].to_list())
            groups.append(t)

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


def search_best_params(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, feature_cols, n_splits=4):
    gkf = GroupKFold(n_splits=n_splits)

    param_candidates = [
        {
            "num_leaves": 31,
            "max_depth": -1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 0.0,
            "reg_alpha": 0.0,
        },
        {
            "num_leaves": 31,
            "max_depth": 7,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
        },
        {
            "num_leaves": 63,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "reg_alpha": 0.1,
        },
    ]

    best_param = None
    best_score = 1e9

    for params in param_candidates:
        fold_maes = []
        for tr_idx, te_idx in gkf.split(X, y, groups):
            X_tr = X.iloc[tr_idx][feature_cols]
            X_te = X.iloc[te_idx][feature_cols]
            y_tr, y_te = y[tr_idx], y[te_idx]

            model = lgb.LGBMRegressor(
                objective="regression",
                learning_rate=0.05,
                n_estimators=500,
                min_child_samples=1,
                min_data_in_bin=1,
                max_bin=255,
                random_state=42,
                **params,
            )
            # ここではverboseは渡さない
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], eval_metric="l1")
            y_pred = model.predict(X_te)
            mae = mean_absolute_error(y_te, y_pred)
            fold_maes.append(mae)

        mean_mae = float(np.mean(fold_maes))
        print(f"[Param trial] {params} -> CV MAE={mean_mae:.4f}")
        if mean_mae < best_score:
            best_score = mean_mae
            best_param = params

    print(f"\nBest params selected: {best_param} (CV MAE={best_score:.4f})")
    return best_param


def run_lgbm_with_shap():
    X, y, groups = build_dataset()

    # 数値化と欠損処理
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.mean(numeric_only=True))

    # 追加特徴量
    X["group_id"] = groups
    X["is_time0"] = (groups == 0).astype(int)

    feature_cols = X.columns.tolist()
    gkf = GroupKFold(n_splits=4)

    # ここで「全ターゲットの結果」をためる入れ物を用意
    metrics_all = []   # 各ターゲットのCV平均を1行ずつ
    results_all = []   # 各サンプル×各foldの予測を1行ずつ

    for tgt in TARGET_COLS:
        print(f"\n===== Target: {tgt} =====")
        y_t = y[tgt].to_numpy()

        fold_maes = []
        fold_rmses = []
        fold_models = []
        fold_valid_idx = []

        # もしハイパーパラメータ探索をするならここで取得しておく
        # best_params = search_best_params(...)
        best_params = {}  # いったん空で

        for fold_id, (tr_idx, te_idx) in enumerate(gkf.split(X, y_t, groups), start=1):
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
                **best_params,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], eval_metric="l1")

            y_pred = model.predict(X_te)

            mae = mean_absolute_error(y_te, y_pred)
            mse = mean_squared_error(y_te, y_pred)
            rmse = np.sqrt(mse)

            fold_maes.append(mae)
            fold_rmses.append(rmse)
            fold_models.append(model)
            fold_valid_idx.append(te_idx)

            # 1サンプルずつ保存
            for i_local, i_global in enumerate(te_idx):
                results_all.append({
                    "target": tgt,
                    "fold": fold_id,
                    "sample_index": int(i_global),
                    "y_true": float(y_te[i_local]),
                    "y_pred": float(y_pred[i_local]),
                    "abs_error": float(abs(y_te[i_local] - y_pred[i_local])),
                })

        # ターゲット単位の集計
        cv_mae_mean = float(np.mean(fold_maes))
        cv_mae_std  = float(np.std(fold_maes))
        cv_rmse_mean = float(np.mean(fold_rmses))
        cv_rmse_std  = float(np.std(fold_rmses))

        print(f"CV MAE:  {cv_mae_mean:.4f} ± {cv_mae_std:.4f}")
        print(f"CV RMSE: {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}")

        # メトリクスをここで1行にまとめておく
        metrics_all.append({
            "target": tgt,
            "cv_mae_mean": cv_mae_mean,
            "cv_mae_std": cv_mae_std,
            "cv_rmse_mean": cv_rmse_mean,
            "cv_rmse_std": cv_rmse_std,
        })

        # 以下はSHAPなど、もとの処理そのまま
        model0 = fold_models[0]
        valid_idx0 = fold_valid_idx[0]
        X_valid0 = X.iloc[valid_idx0][feature_cols]

        explainer = shap.TreeExplainer(model0)
        shap_vals = explainer.shap_values(X_valid0)

        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False)
        shap_df.to_csv(ROOT / f"shap_{tgt}.csv", index=False)

        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_vals, X_valid0, show=False, plot_size=None)
        plt.tight_layout()
        plt.savefig(ROOT / f"shap_{tgt}.png", dpi=200)
        plt.close()

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

    # ここまでで全ターゲットの処理が終わるので、まとめてCSV保存
    metrics_df = pd.DataFrame(metrics_all)
    metrics_df.to_csv(ROOT / "metrics_all_targets.csv", index=False)

    results_df = pd.DataFrame(results_all)
    results_df.to_csv(ROOT / "predictions_all_targets.csv", index=False)

    print("\nAll targets finished.")
    print("Saved: metrics_all_targets.csv, predictions_all_targets.csv")


if __name__ == "__main__":
    run_lgbm_with_shap()
