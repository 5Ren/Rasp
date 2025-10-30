# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.integrate import trapezoid
from scipy.stats import linregress

# ========= パス設定 =========
BASE = Path("/08_Fuji/01_Data_av")

# 出力: 鑑定士評価CSV（正解）
y_path = BASE / "01_appraisers_evaluation.csv"   # 既存の正解ファイル（読み込み）

# 入力: etc（Brix, pH, Conductivity）
etc_path = BASE / "02_etc_data.csv"

# 入力: ORP（味覚センサ）ディレクトリ
orp_dir = BASE / "03_redox_potential"

# 入力: CMOS（匂いセンサ）
cmos_dir = BASE / "04_cmos"

# 入力: QCM（匂いセンサ）
# ご指定ではパス明示がありませんでしたが、05_qcm と仮定します。必要に応じて修正ください。
qcm_dir = BASE / "05_qcm"

# ========= 鑑定士ラベル（ヘッダー） =========
TARGET_COLS = [
    "Degree_of_roasting","Acid_taste","Bitter_taste","Astrigency_mouthfeel",
    "Cup_strength","Body","Aromatic_impact","Floral","Earthy","Tarry","Nutty","Pruney"
]

# ========= 時刻（学習サンプルのキー） =========
TIMES = [0, 2, 4, 6]  # hour

# ========= ユーティリティ =========
def bin_features(x_class: np.ndarray, x_amp: np.ndarray, n_bins: int = 16):
    """Class-Amp スペクトルをビン集約＋代表統計へ変換"""
    if len(x_class) == 0:
        return {}
    cmin, cmax = np.min(x_class), np.max(x_class)
    bins = np.linspace(cmin, cmax, n_bins + 1)
    inds = np.digitize(x_class, bins) - 1
    # ビン平均
    bin_means = []
    for b in range(n_bins):
        sel = x_amp[inds == b]
        bin_means.append(sel.mean() if sel.size > 0 else 0.0)
    bin_means = np.array(bin_means)
    # 代表統計
    f = {
        "orp_amp_sum": float(np.sum(x_amp)),
        "orp_amp_mean": float(np.mean(x_amp)),
        "orp_amp_std": float(np.std(x_amp)),
        "orp_amp_max": float(np.max(x_amp)),
        "orp_class_at_max": float(x_class[np.argmax(x_amp)]),
    }
    # ビン特徴
    for i, v in enumerate(bin_means):
        f[f"orp_bin_mean_{i:02d}"] = float(v)
    return f

def time_series_features(time_s: np.ndarray, values: np.ndarray):
    """時間応答（0-90s程度）から要約特徴を作成"""
    f = {}
    if len(time_s) == 0 or len(values) == 0:
        # 欠損対応
        for k in ["baseline","max","delta","auc","slope_init","t90"]:
            f[k] = np.nan
        return f

    # baseline: 最初の 0-5 s の平均
    base_sel = values[time_s <= 5]
    baseline = float(np.mean(base_sel)) if base_sel.size > 0 else float(values[0])

    vmax = float(np.max(values))
    delta = float(vmax - baseline)

    # 台形積分（AUC）
    auc = float(trapezoid(values, time_s))

    # 初期傾き: 0-10 s を直線近似
    init_sel = (time_s <= 10)
    if np.sum(init_sel) >= 2:
        slope, _, _, _, _ = linregress(time_s[init_sel], values[init_sel])
        slope_init = float(slope)
    else:
        slope_init = np.nan

    # t90: baseline + 0.9*delta に到達する時刻
    target = baseline + 0.9 * delta
    above = np.where(values >= target)[0]
    if delta > 0 and above.size > 0:
        t90 = float(time_s[above[0]])
    else:
        t90 = np.nan

    f.update({
        "baseline": baseline,
        "max": vmax,
        "delta": delta,
        "auc": auc,
        "slope_init": slope_init,
        "t90": t90
    })
    return f

# ========= データ読み込み・特徴量生成 =========
def load_etc():
    df = pd.read_csv(etc_path)
    # 期待ヘッダー: Time[h],Brix,pH,Conductivity
    assert "Time[h]" in df.columns, "02_etc_data.csv に Time[h] 列が必要です"
    return df.set_index("Time[h]").loc[TIMES]

def load_orp():
    # sen_0.csv, sen_2.csv, ...
    rows = []
    for t in TIMES:
        fpath = orp_dir / f"sen_{t}.csv"
        df = pd.read_csv(fpath)
        # 期待ヘッダー: Class, Amp
        x_class = df["Class"].values.astype(float)
        x_amp   = df["Amp"].values.astype(float)
        feats = bin_features(x_class, x_amp, n_bins=16)
        feats["Time[h]"] = t
        rows.append(feats)
    return pd.DataFrame(rows).set_index("Time[h]").loc[TIMES]

def load_cmos():
    # memb_1.csv .. memb_5.csv
    # 期待ヘッダー: Time[s],0,2,4,6
    # 各膜・各時間の要約特徴を連結
    feats_by_time = {t: {} for t in TIMES}
    files = sorted(glob.glob(str(cmos_dir / "memb_*.csv")))
    for f in files:
        mem_id = Path(f).stem.split("_")[-1]
        df = pd.read_csv(f)
        tsec = df["Time[s]"].values.astype(float)
        for t in TIMES:
            vals = df[str(t)].values.astype(float)
            fts = time_series_features(tsec, vals)
            for k, v in fts.items():
                feats_by_time[t][f"cmos_m{mem_id}_{k}"] = v
    # データフレーム化
    rows = []
    for t in TIMES:
        r = {"Time[h]": t}
        r.update(feats_by_time[t])
        rows.append(r)
    return pd.DataFrame(rows).set_index("Time[h]").loc[TIMES]

def load_qcm():
    # gra_01.csv .. gra_16.csv を想定
    # 期待ヘッダー: Time[s],0,2,4,6
    feats_by_time = {t: {} for t in TIMES}
    files = sorted(glob.glob(str(qcm_dir / "gra_*.csv")))
    for f in files:
        mem_id = Path(f).stem.split("_")[-1]
        df = pd.read_csv(f)
        tsec = df["Time[s]"].values.astype(float)
        for t in TIMES:
            vals = df[str(t)].values.astype(float)
            fts = time_series_features(tsec, vals)
            for k, v in fts.items():
                feats_by_time[t][f"qcm_m{mem_id}_{k}"] = v
    rows = []
    for t in TIMES:
        r = {"Time[h]": t}
        r.update(feats_by_time[t])
        rows.append(r)
    return pd.DataFrame(rows).set_index("Time[h]").loc[TIMES]

def load_targets():
    # 正解ラベル
    y = pd.read_csv(y_path)
    assert "Time[h]" in y.columns, "01_appraisers_evaluation.csv に Time[h] 列が必要です"
    # ヘッダー名の正規化（タブや全角スペースの混入対策）
    y.columns = [c.strip() for c in y.columns]
    # 欠けている列があればエラー
    missing = [c for c in TARGET_COLS if c not in y.columns]
    if missing:
        raise ValueError(f"鑑定士評価の列が不足しています: {missing}")
    y = y.set_index("Time[h]").loc[TIMES, TARGET_COLS]
    return y

def build_feature_table():
    etc = load_etc()
    orp = load_orp()
    cmos = load_cmos()
    qcm = load_qcm()
    # インデックス(Time[h])で結合
    X = etc.join(orp, how="inner").join(cmos, how="inner").join(qcm, how="inner")
    return X

# ========= 学習・評価・保存 =========
def train_and_evaluate(X, y):
    # パイプライン
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", MultiOutputRegressor(SVR(kernel="rbf")))
    ])
    # マルチ出力の下位推定器に対するパラメータ指定
    param_grid = {
        "svm__estimator__C": [1, 10, 100],
        "svm__estimator__gamma": ["scale", 0.1, 0.01],
        "svm__estimator__epsilon": [0.1, 0.2],
    }
    # 4サンプルを4分割CV（各時間を1サンプルとみなす）
    # 実用上はサンプルを増やすことが望ましい
    cv = KFold(n_splits=4, shuffle=False)
    gscv = GridSearchCV(pipe, param_grid, cv=cv, scoring="r2", n_jobs=-1, refit=True)
    gscv.fit(X, y)

    # CVのベスト情報
    print("Best params:", gscv.best_params_)
    print("Best CV R2:", gscv.best_score_)

    # ベストモデルで訓練データに対する適合度を確認
    y_pred = gscv.predict(X)
    r2 = r2_score(y, y_pred, multioutput="raw_values")  # 各出力ごと
    mae = np.mean(np.abs(y - y_pred), axis=0)

    print("\nPer-target R2 and MAE:")
    for name, r2i, maei in zip(TARGET_COLS, r2, mae):
        print(f"{name:24s}  R2={r2i:6.3f}  MAE={maei:6.3f}")

    return gscv.best_estimator_

def save_predictions(model, X, out_path):
    pred = model.predict(X)
    df_out = pd.DataFrame(pred, columns=TARGET_COLS, index=X.index)
    df_out = df_out.reset_index()  # Time[h] を列に戻す
    # 指定形式に整形
    df_out = df_out[["Time[h]"] + TARGET_COLS]
    df_out.to_csv(out_path, index=False)
    print(f"\nPredictions saved to: {out_path}")

if __name__ == "__main__":
    # 特徴量とターゲットの作成
    X = build_feature_table()
    y = load_targets()
    # 学習と評価
    model = train_and_evaluate(X, y)

    # 予測CSVの出力先（必要であればパス変更可）
    pred_out = BASE / "02_appraisers_pred_svm.csv"
    save_predictions(model, X, pred_out)
