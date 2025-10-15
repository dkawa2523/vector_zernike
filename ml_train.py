"""
ml_train.py
------------

集約された係数データと条件テーブルから機械学習モデルを構築するスクリプトです。

このスクリプトは、``batch_preprocess.py`` によって生成された
``<dataset_dir>/aggregated/agg_<model_name>.csv`` を読み込み、条件表
``<dataset_dir>/data_table.csv``と結合して説明変数と目的変数を作成し、
単純な線形回帰（最小二乗法）による多目的予測モデルを学習します。

学習後、テストデータに対する平均二乗誤差を表示し、学習した回帰係数
および列名情報をJSON形式で保存します。

使用方法：
    python ml_train.py <dataset_dir> <model_name>

例：
    python ml_train.py dataset vz

上記コマンドは ``dataset/aggregated/agg_vz.csv`` を利用し、
``dataset/model_vz.json`` に係数を保存します。
"""

from __future__ import annotations

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path


def load_aggregated(dataset_dir: Path, model_name: str) -> pd.DataFrame:
    agg_path = dataset_dir / "aggregated" / f"agg_{model_name}.csv"
    if not agg_path.exists():
        raise FileNotFoundError(f"aggregated file not found: {agg_path}")
    return pd.read_csv(agg_path)


def train_linear_regression(X: np.ndarray, Y: np.ndarray):
    """多目的線形回帰モデルを最小二乗法で学習し、係数行列を返す。"""
    # バイアス項を追加
    X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
    # 特徴次元+1 × 目的次元 の係数行列を求める
    B, *_ = np.linalg.lstsq(X_aug, Y, rcond=None)
    return B  # shape: (n_features+1, n_targets)


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: python ml_train.py <dataset_dir> <model_name>")
    dataset_dir = Path(sys.argv[1]).resolve()
    model_name = sys.argv[2]

    # 読み込み
    agg_df = load_aggregated(dataset_dir, model_name)
    feat_df = pd.read_csv(dataset_dir / "data_table.csv")

    # 特徴量列（condition_id以外）
    feature_cols = [c for c in feat_df.columns if c != "condition_id"]
    # 目的変数列（feature_colsとcondition_id以外）
    target_cols = [c for c in agg_df.columns if c not in ("condition_id",) + tuple(feature_cols)]
    if not target_cols:
        raise ValueError("aggregated fileには係数列が含まれていません")

    # 説明変数Xと目的変数Y
    X = agg_df[feature_cols].values
    Y = agg_df[target_cols].values
    n_samples = X.shape[0]
    # シャッフルして学習/テストに分割
    idx = np.arange(n_samples)
    np.random.seed(0)
    np.random.shuffle(idx)
    split = int(0.8 * n_samples) if n_samples > 1 else n_samples
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    # 学習
    B = train_linear_regression(X_train, Y_train)
    # テスト誤差
    X_test_aug = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    Y_pred = X_test_aug @ B
    mse = float(np.mean((Y_test - Y_pred) ** 2)) if len(test_idx) > 0 else float(np.mean((Y_train - (np.hstack([X_train, np.ones((X_train.shape[0],1))]) @ B))**2))
    print(f"model: {model_name}, samples: {n_samples}, mse: {mse:.4f}")
    # モデル保存
    model_info = {
        "B": B.tolist(),
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "mse": mse,
    }
    model_path = dataset_dir / f"model_{model_name}.json"
    with open(model_path, "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"model saved to {model_path}")


if __name__ == "__main__":
    main()
