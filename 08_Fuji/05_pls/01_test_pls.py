# -*- coding: utf-8 -*-
import os
from pathlib import Path
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, r2_score

from scipy.integrate import trapezoid
from scipy.stats import linregress


# =========================================================
# 0. パスなどの定義
# =========================================================
BASE = Path("/Users/ren/PycharmProjects/Rasp/08_Fuji/02_Data_all")

PATH_Y   = BASE / "01_appraisers_evaluation.csv"
PATH_ETC = BASE / "02_etc_data.csv"
DIR_ORP  = BASE / "03_redox_potential"
DIR_CMOS = BASE / "04_cmos"
DIR_QCM  = BASE / "05_qcm"  # 実パスに合わせてください
ROOT = Path(__file__).resolve().parent

TIMES = [0, 2, 4, 6]

TARGET_COLS = [
    "Degree_of_roasting","Acid_taste","Bitter_taste","Astrigency_mouthfeel",
    "Cup_strength","Body","Aromatic_impact","Floral","Earthy","Tarry","Nutty","Pruney"
]


# =========================================================
# 1. 時系列→特徴量
# =========================================================
def extract_ts_features(df: pd.DataFrame, time_col="Time[s]"):
    df = df.copy()
    df = df[df[time_col] >= 30]
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

    # baseline: 30〜35 s
    base_mask = (t >= 30) & (t <= 35)
    if base_mask.any():
        baseline = float(v[base_mask].mean())
    else:
        baseline = float(v[0])

    vmax = float(v.max())
    delta = float(vmax - baseline)
    auc = float(trapezoid(v, t))

    slope_mask = (t >= 30) & (t <= 40)
    if slope_mask.sum() >= 2:
        slope, _, _, _, _ = linregress(t[slope_mask], v[slope_mask])
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
# 2. 入力CSVの読み込み
# =========================================================
def load_etc():
    df = pd.read_csv(PATH_ETC)
    df = df.set_index("Time[h]").loc[TIMES]
    return df


def orp_features_from_df(df: pd.DataFrame, prefix="orp"):
    classes = df["Class"].to_numpy()
    n_cols = [c for c in df.columns if c != "Class"]

    amps = df[n_cols].to_numpy(dtype=float)
    amp_mean = amps.mean(axis=1)
    feats_mean = _orp_to_bins(classes, amp_mean, prefix=prefix)
    feats_list = [("mean", feats_mean)]

    for c in n_cols:
        amp = df[c].to_numpy(dtype=float)
        feats = _orp_to_bins(classes, amp, prefix=prefix)
        feats_list.append((c, feats))

    return feats_list


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
    return orp_features_from_df(df, prefix="orp")


def load_cmos_for_time(t: int):
    pattern = str(DIR_CMOS / f"{t}h_N*.csv")
    files = sorted(glob.glob(pattern))
    all_feats_per_file = []
    for fp in files:
        df = pd.read_csv(fp)
        tsec = df["Time[s]"].to_frame()
        feats_this_file = {}
        for m in range(1, 6):
            col = f"Membrane{m}"
            sub = pd.concat([tsec, df[col]], axis=1)
            sub.columns = ["Time[s]", "Y"]
            fts = extract_ts_features(sub)
            for k, v in fts.items():
                feats_this_file[f"cmos_m{m}_{k}"] = v
        all_feats_per_file.append((Path(fp).stem, feats_this_file))

    if len(all_feats_per_file) == 0:
        return [("mean", {})]

    all_keys = set()
    for _, d in all_feats_per_file:
        all_keys.update(d.keys())

    mean_feats = {}
    for k in all_keys:
        vals = [d.get(k, np.nan) for _, d in all_feats_per_file]
        mean_feats[k] = float(np.nanmean(vals))
    rows = [("mean", mean_feats)]
    rows.extend(all_feats_per_file)
    return rows


def load_qcm_for_time(t: int):
    pattern = str(DIR_QCM / f"{t}h_N*")
    folders = sorted(glob.glob(pattern))
    all_rows = []
    for folder in folders:
        folder_name = Path(folder).name
        graphs = sorted(glob.glob(os.path.join(folder, "graph_*.csv")))
        feats_this_rep = {}
        for gpath in graphs:
            gname = Path(gpath).stem
            df = pd.read_csv(gpath)
            fts = extract_ts_features(df, time_col="Time[s]")
            for k, v in fts.items():
                feats_this_rep[f"qcm_{gname}_{k}"] = v
        all_rows.append((folder_name, feats_this_rep))

    if len(all_rows) == 0:
        return [("mean", {})]

    all_keys = set()
    for _, d in all_rows:
        all_keys.update(d.keys())
    mean_feats = {}
    for k in all_keys:
        vals = [d.get(k, np.nan) for _, d in all_rows]
        mean_feats[k] = float(np.nanmean(vals))
    rows = [("mean", mean_feats)]
    rows.extend(all_rows)
    return rows


def load_targets():
    df = pd.read_csv(PATH_Y)
    df = df.set_index("Time[h]").loc[TIMES]
    return df[TARGET_COLS]


# =========================================================
# 3. 拡張データセットの組み立て
# =========================================================
def build_augmented_dataset():
    y_all = load_targets()
    etc = load_etc()

    X_rows = []
    y_rows = []
    groups = []
    row_tags = []

    for t in TIMES:
        etc_row = etc.loc[t].to_dict()

        orp_list = load_orp_for_time(t)
        orp_mean = orp_list[0][1]

        cmos_list = load_cmos_for_time(t)
        cmos_mean = cmos_list[0][1]

        qcm_list = load_qcm_for_time(t)
        qcm_mean = qcm_list[0][1]

        base_feats = {}
        base_feats.update(etc_row)
        base_feats.update(orp_mean)
        base_feats.update(cmos_mean)
        base_feats.update(qcm_mean)

        # base行
        X_rows.append(base_feats)
        y_rows.append(y_all.loc[t].to_list())
        groups.append(t)
        row_tags.append(f"{t}h_base")

        # ORP差し替え
        for name, feats in orp_list[1:]:
            f = dict(base_feats)
            f.update(feats)
            X_rows.append(f)
            y_rows.append(y_all.loc[t].to_list())
            groups.append(t)
            row_tags.append(f"{t}h_orp_{name}")

        # CMOS差し替え
        for name, feats in cmos_list[1:]:
            f = dict(base_feats)
            f.update(feats)
            X_rows.append(f)
            y_rows.append(y_all.loc[t].to_list())
            groups.append(t)
            row_tags.append(f"{t}h_cmos_{name}")

        # QCM差し替え
        for name, feats in qcm_list[1:]:
            f = dict(base_feats)
            f.update(feats)
            X_rows.append(f)
            y_rows.append(y_all.loc[t].to_list())
            groups.append(t)
            row_tags.append(f"{t}h_qcm_{name}")

    X = pd.DataFrame(X_rows)
    y = pd.DataFrame(y_rows, columns=TARGET_COLS)
    groups = np.array(groups)
    return X, y, groups, row_tags


# =========================================================
# 4. PLSで学習・評価
# =========================================================
def train_eval_pls():
    X, y, groups, tags = build_augmented_dataset()

    # 欠損をとりあえず列平均で埋める
    X = X.fillna(X.mean(numeric_only=True))

    # ここで「効いている特徴だけ」にある程度絞る
    keep_cols = []
    for c in X.columns:
        if c.startswith("qcm_graph_12_"):
            keep_cols.append(c)
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
        elif c.startswith("cmos_m4_"):
            keep_cols.append(c)
        elif c in ["pH", "Conductivity", "Brix_1", "Brix_n2"]:
            keep_cols.append(c)
        elif c in ["orp_bin_06", "orp_bin_07", "orp_max", "orp_class_at_max"]:
            keep_cols.append(c)
    # 念のため存在チェック
    keep_cols = [c for c in keep_cols if c in X.columns]
    X_sel = X[keep_cols].copy()

    # 0hフラグも入れておくと安定する
    X_sel["is_time0"] = (groups == 0).astype(int)

    scaler = StandardScaler()

    # 成分数を小さいところで試す
    n_comp_candidates = [2, 3, 4, 5, 6]
    gkf = GroupKFold(n_splits=4)

    best_mae = np.inf
    best_ncomp = None
    best_model = None

    for n_comp in n_comp_candidates:
        fold_maes = []
        for tr_idx, te_idx in gkf.split(X_sel, y, groups):
            X_tr, X_te = X_sel.iloc[tr_idx], X_sel.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_tr_s, y_tr)
            y_pred = pls.predict(X_te_s)

            # グループごとの評価ではなくfold単位でのMAE
            mae = mean_absolute_error(y_te, y_pred)
            fold_maes.append(mae)

        avg_mae = np.mean(fold_maes)
        print(f"n_components={n_comp}: CV MAE={avg_mae:.4f}")
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_ncomp = n_comp

    # ベストで学習し直す
    print(f"\nBest n_components: {best_ncomp} (CV MAE={best_mae:.4f})")
    X_s = scaler.fit_transform(X_sel)
    pls = PLSRegression(n_components=best_ncomp)
    pls.fit(X_s, y)
    y_pred_all = pls.predict(X_s)

    # 時間ごとに平均して評価
    print("\n[Group-level evaluation]")
    group_true = []
    group_pred = []
    for t in TIMES:
        mask = (groups == t)
        y_true_t = y[mask].iloc[0:1].to_numpy()
        y_pred_t = y_pred_all[mask].mean(axis=0, keepdims=True)
        mae_t = mean_absolute_error(y_true_t, y_pred_t)
        print(f"time {t}h: MAE(group-mean)={mae_t:.3f}, n_rows={mask.sum()}")
        group_true.append(y_true_t)
        group_pred.append(y_pred_t)

    y_true_all = np.vstack(group_true)
    y_pred_all2 = np.vstack(group_pred)
    r2_g = r2_score(y_true_all, y_pred_all2, multioutput="variance_weighted")
    print(f"\nR2 over 4 time-points (group-mean): {r2_g:.3f}")

    # 回帰係数の解釈用に保存
    # 修正後
    coef_mat = pls.coef_
    # 必要なら転置する
    if coef_mat.shape == (len(TARGET_COLS), len(keep_cols) + 1):
        coef_mat = coef_mat.T

    coef_df = pd.DataFrame(
        coef_mat,
        index=keep_cols + ["is_time0"],
        columns=TARGET_COLS
    )
    out_path = ROOT / "pls_coefficients.csv"
    coef_df.to_csv(out_path)
    print(f"PLS係数を保存しました: {out_path}")


if __name__ == "__main__":
    train_eval_pls()
