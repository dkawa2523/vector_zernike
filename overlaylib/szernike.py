# overlaylib/szernike.py  ── 追加ファイル ──────────────────────────
import numpy as np, pandas as pd, time
from .core import Decomposer
from .utils import _R                          # 既存 radial 関数を流用
from .overlay_io   import OutputBundle

def Z(n, m, x, y):
    """スカラー Zernike Z_nm (OSA 定義, 非正規化)"""
    r  = np.hypot(x, y)
    th = np.arctan2(y, x)
    Rnm = _R(n, abs(m), r)
    if m == 0:
        return Rnm
    if m > 0:
        return Rnm * np.cos(m * th)
    return Rnm * np.sin(-m * th)

def design_matrix(xn, yn, modes):
    A = np.zeros((len(xn), len(modes)))
    for j, (n, m) in enumerate(modes):
        A[:, j] = Z(n, m, xn, yn)
    return A

class ScalarZernike(Decomposer):
    """
    dx と dy をそれぞれ独立に Zernike 展開。
    coeff DataFrame は columns=["family","n","m","coef"]
       family='X' (dx) / 'Y' (dy)
    """
    def fit(self, df: pd.DataFrame):
        t0 = time.time()
        n_max = self.cfg["n_max"]
        modes = [(n, m) for n in range(n_max + 1)      # n = 0 も含める
                           for m in range(-n, n + 1, 2)]
        A = design_matrix(df.xn.values, df.yn.values, modes)

        # --- dx ---
        coef_x, *_ = np.linalg.lstsq(A, df.dx.values, rcond=None)
        fit_dx = A @ coef_x

        # --- dy ---
        coef_y, *_ = np.linalg.lstsq(A, df.dy.values, rcond=None)
        fit_dy = A @ coef_y

        # --- 係数テーブル結合 ---
        coeff = pd.DataFrame({
            "family": ["X"] * len(modes) + ["Y"] * len(modes),
            "n":      [n for n, _ in modes] + [n for n, _ in modes],
            "m":      [m for _, m in modes] + [m for _, m in modes],
            "coef":   np.concatenate([coef_x, coef_y])
        })

        fit_vec = np.column_stack([fit_dx, fit_dy])
        rms = np.sqrt(np.mean((df[["dx","dy"]].values - fit_vec) ** 2))
        return OutputBundle("sz", coeff, fit_vec, rms,
                            {"fit_sec": time.time() - t0})