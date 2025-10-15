"""
overlay_run.py
----------------

このモジュールはテンプレートYAMLに基づいてデータセットを分解し、
Vector/Complex/Scalar Zernike や多項式オーバーレイの係数推定を実行し、
結果を保存する高水準の関数 ``run_cfg(cfg)`` を提供します。

従来のスクリプトでは ``batch_preprocess.py`` から ``overlay_run.run_cfg``
が呼び出されるだけで、実装はユーザに委ねられていました。本実装では
次の機能を追加しています：

* 測定データから正規化座標 ``xn, yn`` を自動計算する。
  入力CSVに ``xn`` と ``yn`` 列が存在しない場合、ウェハ中心と半径を求めて
  ``xn=(x-x_center)/r_max``, ``yn=(y-y_center)/r_max`` を追加します。

* テンプレートYAMLの柔軟な読み取り。``cfg`` 辞書内に ``models`` 配列が
  存在する場合はその内容を優先し、古いスタイル（``vector_zernike`` 等の
  直接指定）にも対応します。``enable`` フラグが False の場合はその
  モデルをスキップします。

* 出力の細かな制御。``io.precision`` と ``io.save_png`` の設定を
  ``save_bundle`` に渡し、CSVの有効桁数やPNGの生成有無を変更できます。

* 処理結果（各手法ごとの ``OutputBundle`` のリスト）を返すので、
  上位スクリプトが集約データセットを作成したり、機械学習用に利用したり
  できます。

使い方の例：
    >>> import yaml
    >>> from overlay_run import run_cfg
    >>> cfg = yaml.safe_load(open('batch_template.yaml'))
    >>> cfg['global']['data_file'] = 'dataset/test_id01.csv'
    >>> cfg['global']['output_root'] = 'dataset/results_id01'
    >>> bundles = run_cfg(cfg)
    # 各 ``OutputBundle`` は bund.name, bund.coeff, bund.rms などを持つ
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

# ルートディレクトリに配置されている各モジュールを直接インポートする。
# 本リポジトリでは overlaylib というサブパッケージではなく、
# core.py, vzernike.py などがルートに存在するため、相対インポートはせずに
# 直接インポートする。
from .vzernike import VecZernike
from .czernike import CpxZernike
from .szernike import ScalarZernike
from .poly import PolyOverlay
from .overlay_io import save_bundle
from .report import make_report

def _compute_normalized_coords(df: pd.DataFrame) -> pd.DataFrame:
    """x,y列から正規化座標を計算し、``xn`` ``yn`` 列を追加したDataFrameを返す。"""
    # ウェハ中心を算出
    x0 = 0.5 * (df['x'].min() + df['x'].max())
    y0 = 0.5 * (df['y'].min() + df['y'].max())
    # 最大半径
    r_max = np.sqrt(((df['x'] - x0) ** 2 + (df['y'] - y0) ** 2).max())
    if r_max == 0:
        # すべて同一点の場合は半径を1としておく
        r_max = 1.0
    df = df.copy()
    df['xn'] = (df['x'] - x0) / r_max
    df['yn'] = (df['y'] - y0) / r_max
    return df

def _parse_models(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    テンプレートから実行すべきモデル設定のリストを抽出する。
    cfg['models'] が存在すればその内容を返し、存在しなければ旧形式の
    ``vector_zernike``, ``complex_zernike``, ``scalar_zernike``, ``overlay_poly``
    キーを読み取って生成する。
    """
    models: List[Dict[str, Any]] = []
    if 'models' in cfg and isinstance(cfg['models'], list):
        for m in cfg['models']:
            # enable が明示的に False ならスキップ
            if not m.get('enable', True):
                continue
            models.append(m.copy())
        return models
    # 旧形式の解釈
    mapping = [
        ('vector_zernike', 'vzernike'),
        ('complex_zernike', 'czernike'),
        ('scalar_zernike', 'szernike'),
        ('overlay_poly', 'poly'),
    ]
    for key, kind in mapping:
        section = cfg.get(key)
        if not section:
            continue
        if section.get('enable', True) is False:
            continue
        m = {'kind': kind}
        for k, v in section.items():
            if k == 'enable':
                continue
            m[k] = v
        models.append(m)
    return models


def run_cfg(cfg: Dict[str, Any]):
    """
    デコンポジションを実行し、各モデルの ``OutputBundle`` を返す。

    Parameters
    ----------
    cfg : dict
        テンプレートYAMLを読み込んだ辞書。少なくとも ``global`` セクション
        に ``data_file`` と ``output_root`` が含まれている必要があります。

    Returns
    -------
    List[overlaylib.io.OutputBundle]
        実行したモデルごとに得られた ``OutputBundle`` のリスト。
    """
    g = cfg.get('global', {})
    data_file = g.get('data_file')
    out_root = g.get('output_root')
    if not data_file or not out_root:
        raise ValueError('cfg["global"]["data_file"] と cfg["global"]["output_root"] を指定してください')
    df = pd.read_csv(data_file)
    # xn, yn の自動生成
    if 'xn' not in df.columns or 'yn' not in df.columns:
        df = _compute_normalized_coords(df)
    # モデル設定取得
    models = _parse_models(cfg)
    if not models:
        raise ValueError('実行するモデルが1つも定義されていません')
    # 出力ディレクトリ作成
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    # save_bundle に渡すオプション
    io_cfg = cfg.get('io', {})
    precision = io_cfg.get('precision', 8)
    save_png = io_cfg.get('save_png', True)
    # モデル実行
    bundles = []
    for m in models:
        kind = m.get('kind', '').lower()
        # Vector Zernike
        if kind in ('vzernike', 'vector_zernike', 'vz'):
            dec = VecZernike({'n_max': m.get('n_max', 4), 'include_B': m.get('include_B', True)})
            b = dec.fit(df)
            bundles.append(b)
            save_bundle(b, df, root, precision=precision, save_png=save_png)
        # Complex Zernike
        elif kind in ('czernike', 'complex_zernike', 'cz'):
            dec = CpxZernike({'n_max': m.get('n_max', 4)})
            b = dec.fit(df)
            bundles.append(b)
            save_bundle(b, df, root, precision=precision, save_png=save_png)
        # Scalar Zernike
        elif kind in ('szernike', 'scalar_zernike', 'sz'):
            dec = ScalarZernike({'n_max': m.get('n_max', 4)})
            b = dec.fit(df)
            bundles.append(b)
            save_bundle(b, df, root, precision=precision, save_png=save_png)
        # Polynomial overlay
        elif kind in ('poly', 'overlay_poly', 'poly_overlay'):
            terms = m.get('model_terms')
            if not terms:
                raise ValueError('poly overlay のmodel_termsが指定されていません')
            dec = PolyOverlay({'model_terms': terms})
            poly_bundles = dec.fit(df)
            for b in poly_bundles:
                bundles.append(b)
                save_bundle(b, df, root, precision=precision, save_png=save_png)
        else:
            raise ValueError(f'未知のモデル種別: {kind}')
    # HTMLレポート
    try:
        make_report(bundles, root / 'summary.html')
    except Exception:
        # レポート生成に失敗しても処理は続行
        pass
    return bundles
