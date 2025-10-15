#!/usr/bin/env python
"""
batch_preprocess.py
-------------------

テンプレートYAMLに基づき、測定データ（test_<condition_id>.csv）を一括展開し、
各モデルの係数を横持ち形式のCSVに集約して保存するバッチスクリプトです。

機能概要:
  * YAMLのglobal.dataset_dirにデータセットディレクトリを指定しておけば、
    コマンドライン引数を省略して実行できます。
  * global.aggregated_dirに集約CSV出力先を指定できます。
    （絶対パスならそのまま、相対パスならdataset_dirからの相対）
  * 個々の条件に対してoverlaylib.overlay_run.run_cfg()を呼び出し、
    results_<condition_id>/ ディレクトリに展開結果を保存します。
  * 全ての条件の係数を結合し、aggregated/agg_<model>.csvに保存します。

使用例:

    python batch_preprocess.py                   # YAMLに従って実行
    python batch_preprocess.py data_dir          # data_dirを優先して実行
    python batch_preprocess.py data_dir config.yaml

"""

import sys, copy, yaml
from pathlib import Path
import pandas as pd

from overlaylib.overlay_run import run_cfg


def load_template(yaml_path: Path) -> dict:
    """YAMLファイルを読み込んで辞書を返す。存在しなければエラー。"""
    if not yaml_path.exists():
        sys.exit(f"[ERROR] template YAML not found: {yaml_path}")
    return yaml.safe_load(yaml_path.read_text())


def _extract_yaml_path(argv) -> Path:
    """コマンドライン引数からテンプレートYAMLのパスを決定する。"""
    # argv[1] が YAMLならそれを、次に argv[2] を参照。なければ default。
    if len(argv) >= 2 and argv[1].lower().endswith(('.yaml', '.yml')):
        return Path(argv[1])
    if len(argv) >= 3 and argv[2].lower().endswith(('.yaml', '.yml')):
        return Path(argv[2])
    return Path('batch_template.yaml')


def _extract_dataset_dir(argv, cfg) -> Path:
    """コマンドライン引数またはYAMLから dataset_dir を決定する。"""
    # 引数に data_dir があればそれを優先（ただし YAML と区別）
    if len(argv) >= 2 and not argv[1].lower().endswith(('.yaml', '.yml')):
        return Path(argv[1]).resolve()
    ds = cfg.get('global', {}).get('dataset_dir')
    if ds:
        return Path(ds).resolve()
    sys.exit('[ERROR] dataset_dir must be specified via CLI or YAML global.dataset_dir')


def _coeff_to_dict(bundle) -> dict[str, float]:
    """
    OutputBundle の coeff DataFrame を辞書に変換するユーティリティ。
    ベクトル/スカラー Zernike は family_n_m, 複素Zernikeは Re/Im_n_m,
    多項式モデルは param名のままとします。
    """
    df = bundle.coeff
    name = bundle.name.lower()
    d: dict[str, float] = {}
    if name == 'vz':
        # family,n,m,coef
        for fam, n, m, c in df[['family','n','m','coef']].itertuples(index=False):
            d[f"{fam}_{int(n)}_{int(m)}"] = float(c)
    elif name == 'cz':
        # n,m,Re,Im
        for n, m, re, im in df[['n','m','Re','Im']].itertuples(index=False):
            d[f"Re_{int(n)}_{int(m)}"] = float(re)
            d[f"Im_{int(n)}_{int(m)}"] = float(im)
    elif name == 'sz':
        # family,n,m,coef
        for fam, n, m, c in df[['family','n','m','coef']].itertuples(index=False):
            d[f"{fam}_{int(n)}_{int(m)}"] = float(c)
    else:
        # polynomial: param, coef
        for param, c in df[['param','coef']].itertuples(index=False):
            d[str(param)] = float(c)
    return d


def main():
    argv = sys.argv
    tpl_path = _extract_yaml_path(argv)
    cfg = load_template(tpl_path)
    data_root = _extract_dataset_dir(argv, cfg)

    # 条件テーブル読み込み
    cond_path = data_root / 'data_table.csv'
    if not cond_path.exists():
        sys.exit(f"[ERROR] data_table.csv not found in {data_root}")
    cond_df = pd.read_csv(cond_path)
    if 'condition_id' not in cond_df.columns:
        sys.exit("[ERROR] data_table.csv must contain 'condition_id'")
    condition_ids = cond_df['condition_id'].astype(str).tolist()

    # 係数集約用ディクショナリ model_name -> list of rows
    aggregated: dict[str, list[dict[str, float]]] = {}

    for cid in condition_ids:
        csv_file = data_root / f"test_{cid}.csv"
        if not csv_file.exists():
            print(f"[WARN] missing {csv_file.name} … skip")
            continue
        # 個別設定を書き換え
        cfg_local = copy.deepcopy(cfg)
        g = cfg_local.setdefault('global', {})
        g['data_file'] = str(csv_file)
        g['output_root'] = str(data_root / f"results_{cid}")
        try:
            bundles = run_cfg(cfg_local)
        except Exception as e:
            print(f"[ERROR] run_cfg failed for {cid}: {e}")
            continue
        # 各モデルの係数を収集
        for b in bundles:
            row = {'condition_id': cid}
            row.update(_coeff_to_dict(b))
            aggregated.setdefault(b.name, []).append(row)

    # 集約ディレクトリ決定
    agg_root = cfg.get('global', {}).get('aggregated_dir')
    if agg_root:
        p = Path(agg_root)
        agg_dir = p if p.is_absolute() else (data_root / p)
    else:
        agg_dir = data_root / 'aggregated'
    agg_dir.mkdir(parents=True, exist_ok=True)

    # 各モデルの集約CSV出力
    for model_name, rows in aggregated.items():
        if not rows:
            continue
        df_coeff = pd.DataFrame(rows)
        merged = pd.merge(cond_df, df_coeff, on='condition_id', how='left')
        out_path = agg_dir / f"agg_{model_name}.csv"
        merged.to_csv(out_path, index=False)
        print(f"[INFO] aggregated coefficients for {model_name} -> {out_path}")


if __name__ == '__main__':
    main()