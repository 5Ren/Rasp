# -*- coding: utf-8 -*-
import os
from pathlib import Path
import glob
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.integrate import trapezoid
from scipy.stats import linregress

"""
/Users/ren/PycharmProjects/Rasp/08_Fuji/02_Data_all/
  ├── 01_appraisers_evaluation.csv
  ├── 02_etc_data.csv
  ├── 03_redox_potential/
  │     ├── sen_0.csv   # Class,N1,N2,N3
  │     ├── sen_2.csv
  │     ├── sen_4.csv
  │     └── sen_6.csv
  ├── 04_cmos/
  │     ├── 0h_N1.csv   # Time[s],Membrane1..5
  │     ├── 0h_N2.csv
  │     ├── 0h_N3.csv
  │     ├── 2h_N1.csv
  │     ...
  │     └── 6h_N3.csv
  └── 05_qcm/  
        ├── 0h_N1/
        │     ├── graph_01.csv   # Time[s],Y
        │     ├── ...
        │     └── graph_22.csv
        ├── 0h_N2/
        ...
        └── 6h_N5/

"""

# =========================================================
# 1. パス定義
# =========================================================
BASE = Path("/Users/ren/PycharmProjects/Rasp/08_Fuji/02_Data_all")

PATH_Y   = BASE / "01_appraisers_evaluation.csv"
PATH_ETC = BASE / "02_etc_data.csv"
DIR_ORP  = BASE / "03_redox_potential"
DIR_CMOS = BASE / "04_cmos"
DIR_QCM  = BASE / "05_qcm"   # ←ここだけ環境に合わせて変えてください

TIMES = [0, 2, 4, 6]

TARGET_COLS = [
    "Degree_of_roasting","Acid_taste","Bitter_taste","Astrigency_mouthfeel",
    "Cup_strength","Body","Aromatic_impact","Floral","Earthy","Tarry","Nutty","Pruney"
]

# =========================================================
# 2. 共通の時間応答→特徴量
#    CMOS/QCMはTime[s] >= 30だけを使う
# =========================================================
def extract_ts_features(df: pd.DataFrame, time_col="Time[s]"):
    """Time[s]列と値1列のデータフレームを受け取り、30s以降から特徴量を取る"""
    df = df.copy()
    df = df[df[time_col] >= 30]     # 30s以降のみ有効
    t = df[time_col].to_numpy()
    v = df.iloc[:, 1].to_numpy()    # 2列目が値

    if len(t) == 0:
        return {
            "baseline_30s": np.nan,
            "max_30s": np.nan,
            "delta_30s": np.nan,
            "auc_30s": np.nan,
            "slope_30s": np.nan,
        }

    # 基線: 30〜35sの平均
    base_mask = (t >= 30) & (t <= 35)
    baseline = float(v[base_mask].mean()) if base_mask.any() else float(v[0])

    vmax = float(v.max())
    delta = float(vmax - baseline)
    auc = float(trapezoid(v, t))

    # 30〜40sの傾き
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
# 3. etcの読み込み (Brix_1, Brix_n2, pH, Conductivity)
# =========================================================
def load_etc():
    df = pd.read_csv(PATH_ETC)
    # 期待: Time[h],Brix_1,Brix_n2,pH,Conductivity
    df = df.set_index("Time[h]").loc[TIMES]
    return df

# =========================================================
# 4. ORPの読み込み
#    sen_{t}.csv: Class,N1,N2,N3
#    → base(平均) + 各Nをレプリケートとして返す
# =========================================================
def orp_features_from_df(df: pd.DataFrame, prefix="orp"):
    # df: Class, N1, N2, N3 の想定
    classes = df["Class"].to_numpy()
    feats_list = []

    # N列を列挙（N1..N3のうち存在する分だけ）
    n_cols = [c for c in df.columns if c != "Class"]

    # まず平均を作る
    amps = df[n_cols].to_numpy(dtype=float)
    amp_mean = amps.mean(axis=1)
    feats_mean = _orp_to_bins(classes, amp_mean, prefix=prefix)
    feats_list.append(("mean", feats_mean))

    # 各レプリケート
    for c in n_cols:
        amp = df[c].to_numpy(dtype=float)
        feats = _orp_to_bins(classes, amp, prefix=prefix)
        feats_list.append((c, feats))

    return feats_list   # [("mean",{...}),("N1",{...}),...]

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

# =========================================================
# 5. CMOSの読み込み
#    /04_cmos/0h_N1.csv ... 6h_N3.csv
#    1ファイルに5膜分の波形があるので、膜ごとに特徴量を作って統合
# =========================================================
def load_cmos_for_time(t: int):
    # 0h,2h,4h,6h のフォーマットを作る
    pattern = str(DIR_CMOS / f"{t}h_N*.csv")
    files = sorted(glob.glob(pattern))
    # ベースは「同時刻の全ファイルの平均」を作る
    # そのうえで「各ファイルを差し替えた行」を作る
    # まず全ファイル分をパース
    all_feats_per_file = []
    for fp in files:
        df = pd.read_csv(fp)
        # Time[s],Membrane1..5
        tsec = df["Time[s]"].to_frame()
        feats_this_file = {}
        for m in range(1, 6):
            col = f"Membrane{m}"
            sub = pd.concat([tsec, df[col]], axis=1)
            sub.columns = ["Time[s]", "Y"]
            fts = extract_ts_features(sub)
            # キーに膜番号とレプリケート番号を埋め込む
            for k, v in fts.items():
                feats_this_file[f"cmos_m{m}_{k}"] = v
        all_feats_per_file.append((Path(fp).stem, feats_this_file))

    # 平均をとる
    if len(all_feats_per_file) == 0:
        return [("mean", {})]

    # 全キーを列挙
    all_keys = set()
    for _, d in all_feats_per_file:
        all_keys.update(d.keys())

    mean_feats = {}
    for k in all_keys:
        vals = [d.get(k, np.nan) for _, d in all_feats_per_file]
        mean_feats[k] = float(np.nanmean(vals))
    rows = [("mean", mean_feats)]

    # 各ファイルを個別行として返す
    for name, d in all_feats_per_file:
        rows.append((name, d))

    return rows   # [("mean",{...}),("0h_N1",{...}),...]

# =========================================================
# 6. QCMの読み込み
#    /05_qcm/0h_N1/graph_01.csv ...
#    30s以降で各graphの特徴量→全部まとめて1行
# =========================================================
def load_qcm_for_time(t: int):
    pattern = str(DIR_QCM / f"{t}h_N*")
    folders = sorted(glob.glob(pattern))
    all_rows = []

    for folder in folders:
        folder_name = Path(folder).name   # 例: 0h_N3
        # 中の graph_*.csv を読む
        graphs = sorted(glob.glob(os.path.join(folder, "graph_*.csv")))
        feats_this_rep = {}
        for gpath in graphs:
            gname = Path(gpath).stem      # graph_01
            df = pd.read_csv(gpath)
            # 期待: Time[s],Y
            fts = extract_ts_features(df, time_col="Time[s]")
            for k, v in fts.items():
                feats_this_rep[f"qcm_{gname}_{k}"] = v
        all_rows.append((folder_name, feats_this_rep))

    if len(all_rows) == 0:
        return [("mean", {})]

    # 平均行を作る
    all_keys = set()
    for _, d in all_rows:
        all_keys.update(d.keys())
    mean_feats = {}
    for k in all_keys:
        vals = [d.get(k, np.nan) for _, d in all_rows]
        mean_feats[k] = float(np.nanmean(vals))
    rows = [("mean", mean_feats)]
    rows.extend(all_rows)
    return rows   # [("mean",{...}),("0h_N1",{...}),...]

# =========================================================
# 7. ラベルの読み込み
# =========================================================
def load_targets():
    df = pd.read_csv(PATH_Y)
    df = df.set_index("Time[h]").loc[TIMES]
    # 必要な列だけ
    return df[TARGET_COLS]

# =========================================================
# 8. データセット拡張
#    各時刻で
#      1) etc(1行) + orp_mean + cmos_mean + qcm_mean → base
#      2) orpの各Nで置換した行を追加
#      3) cmosの各Nで置換した行を追加
#      4) qcmの各Nで置換した行を追加
# =========================================================
def build_augmented_dataset():
    y_all = load_targets()
    etc = load_etc()

    X_rows = []
    y_rows = []
    groups = []
    row_tags = []

    for t in TIMES:
        # etc
        etc_row = etc.loc[t].to_dict()

        # orp
        orp_list = load_orp_for_time(t)     # [("mean", {...}), ("N1", {...}), ...]
        orp_mean = orp_list[0][1]

        # cmos
        cmos_list = load_cmos_for_time(t)
        cmos_mean = cmos_list[0][1]

        # qcm
        qcm_list = load_qcm_for_time(t)
        qcm_mean  = qcm_list[0][1]

        # 1) base行
        base_feats = {}
        base_feats.update(etc_row)
        base_feats.update(orp_mean)
        base_feats.update(cmos_mean)
        base_feats.update(qcm_mean)

        X_rows.append(base_feats)
        y_rows.append(y_all.loc[t].to_list())
        groups.append(t)
        row_tags.append(f"{t}h_base")

        # 2) ORPだけ差し替え
        for name, feats in orp_list[1:]:
            f = dict(base_feats)
            for k, v in feats.items():
                f[k] = v
            X_rows.append(f)
            y_rows.append(y_all.loc[t].to_list())
            groups.append(t)
            row_tags.append(f"{t}h_orp_{name}")

        # 3) CMOSだけ差し替え
        for name, feats in cmos_list[1:]:
            f = dict(base_feats)
            for k, v in feats.items():
                f[k] = v
            X_rows.append(f)
            y_rows.append(y_all.loc[t].to_list())
            groups.append(t)
            row_tags.append(f"{t}h_cmos_{name}")

        # 4) QCMだけ差し替え
        for name, feats in qcm_list[1:]:
            f = dict(base_feats)
            for k, v in feats.items():
                f[k] = v
            X_rows.append(f)
            y_rows.append(y_all.loc[t].to_list())
            groups.append(t)
            row_tags.append(f"{t}h_qcm_{name}")

    X = pd.DataFrame(X_rows)
    y = pd.DataFrame(y_rows, columns=TARGET_COLS)
    groups = np.array(groups)
    return X, y, groups, row_tags

# =========================================================
# 9. 学習・評価(GroupKFoldで4分割、scoring=MAE)
# =========================================================
def train_and_eval():
    X, y, groups, tags = build_augmented_dataset()

    # 欠損がもし入ったら一旦0埋めや平均埋めにしておく
    X = X.fillna(X.mean(numeric_only=True))

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", MultiOutputRegressor(SVR(kernel="rbf")))
    ])

    param_grid = {
        "svm__estimator__C": [1, 10, 100],
        "svm__estimator__gamma": ["scale", 0.1],
        "svm__estimator__epsilon": [0.1, 0.2],
    }

    gkf = GroupKFold(n_splits=4)
    gscv = GridSearchCV(
        pipe,
        param_grid,
        cv=gkf.split(X, y, groups=groups),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=True
    )
    gscv.fit(X, y)

    print("Best params:", gscv.best_params_)
    print("Best CV MAE:", -gscv.best_score_)

    # 各foldの外部予測を手で集計したければここで回す
    # ここではベストモデルで全行を予測しておく
    y_pred = gscv.predict(X)

    # グループ単位でのR2/MAEをざっくり見る
    for t in TIMES:
        mask = (groups == t)
        r2 = r2_score(y[mask], y_pred[mask], multioutput="variance_weighted")
        mae = mean_absolute_error(y[mask], y_pred[mask])
        print(f"time {t}h: R2={r2:.3f}, MAE={mae:.3f}, n_rows={mask.sum()}")

    # 予測を保存（任意）
    out_pred = BASE / "02_appraisers_pred_svm.csv"
    out_df = pd.DataFrame(y_pred, columns=TARGET_COLS)
    out_df.insert(0, "group_Time[h]", groups)
    out_df.insert(1, "row_tag", tags)
    out_df.to_csv(out_pred, index=False)
    print(f"saved: {out_pred}")

if __name__ == "__main__":
    train_and_eval()
