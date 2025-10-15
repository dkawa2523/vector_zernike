from dataclasses import dataclass
from pathlib import Path
import numpy as np, pandas as pd, json, matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# ========= dataclass =================================================
@dataclass
class OutputBundle:
    name: str               # "vz", "cz", "poly26" …
    coeff: pd.DataFrame     # 各手法ごとフォーマットが異なる
    fit_vec: np.ndarray     # (K,2)
    rms: float
    timings: dict

# ========= 軽量メタを書き出す関数 ====================================
def _write_meta(b: "OutputBundle", path: Path) -> None:
    meta = {
        "name": b.name,
        "rms_nm": b.rms,
        **b.timings,
        "n_coeff": len(b.coeff),
        "n_points": int(b.fit_vec.shape[0]),
    }
    path.write_text(json.dumps(meta, indent=2))

# ========= 係数ヒストグラム ==========================================
def _plot_coeff_hist(bundle: "OutputBundle", out_png: Path) -> None:
    """
    bundle.coeff の内容を自動判別してカテゴリカル棒グラフを作成
      • Vector-Zernike : family,n,m,coef
      • Complex-Zernike: n,m,Re,Im  → 2 枚呼び出し
      • Poly           : param,coef
    """
    df = bundle.coeff

    # ---- Complex Zernike (Re / Im) ----------------------------------
    if {"Re","Im"}.issubset(df.columns):
        for part in ("Re", "Im"):
            labels = [f"{n},{m}" for n,m in zip(df.n, df.m)]
            values = df[part]
            _bar_plot(labels, values,
                      f"{bundle.name}  {part} coefficients",
                      out_png.with_name(f"hist_{bundle.name}_{part}.png"))
        return

    # ---- Vector Zernike --------------------------------------------
    if {"family","n","m","coef"}.issubset(df.columns):
        labels = [f"{fam}_{n},{m}" for fam,n,m in
                  zip(df.family, df.n, df.m)]
        _bar_plot(labels, df.coef,
                  f"{bundle.name} coefficients",
                  out_png)
        return

    # ---- Polynomial -------------------------------------------------
    if {"param","coef"}.issubset(df.columns):
        _bar_plot(df.param, df.coef,
                  f"{bundle.name} coefficients",
                  out_png)
        return

    raise ValueError("Unrecognized coeff table format")

def _bar_plot(labels, values, title, out_png: Path):
    fig = plt.figure(figsize=(0.6*len(labels)+1, 4))
    plt.bar(range(len(values)), values, edgecolor="k")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
    plt.ylabel("coefficient [nm]")
    plt.title(title)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

# ========= フィット vs 生データ散布図 ===============================
def _plot_scatter(raw, fit, title, out_png):
    # 線形回帰
    A = np.vstack([raw, np.ones_like(raw)]).T
    slope, intercept = np.linalg.lstsq(A, fit, rcond=None)[0]
    pred = slope * raw + intercept
    mse  = np.mean((fit - raw)**2)
    ss_tot = np.sum((raw - raw.mean())**2)
    ss_res = np.sum((fit - raw)**2)
    r2 = 1 - ss_res / ss_tot if ss_tot else 0

    fig = plt.figure(figsize=(5,5))
    plt.scatter(raw, fit, s=8, alpha=0.6)
    xmin, xmax = plt.gca().get_xlim()
    ymin, ymax = plt.gca().get_ylim()
    lo = min(xmin, ymin); hi = max(xmax, ymax)
    plt.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal y=x")
    plt.plot(raw, pred, "r-", lw=1,
             label=f"fit  y={slope:.3f}x+{intercept:.3f}")
    plt.xlabel("raw [nm]");  plt.ylabel("fit [nm]")
    plt.title(title)
    plt.legend(loc="upper left", fontsize=8)
    txt = f"slope = {slope:.3f}\nR²    = {r2:.4f}\nMSE  = {mse:.2f}"
    plt.text(0.02, 0.80, txt, transform=plt.gca().transAxes,
             fontsize=8, bbox=dict(fc="w", ec="k", alpha=0.6))
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

# ========= メイン保存関数 ===========================================
def save_bundle(bundle: "OutputBundle",
                df_raw: pd.DataFrame,
                root: Path,
                precision: int = 8,
                save_png: bool = True) -> None:

    tag = bundle.name
    out = root / tag
    out.mkdir(parents=True, exist_ok=True)

    # ---- CSV & JSON -----------------------------------------------
    bundle.coeff.to_csv(out/f"coeff_{tag}.csv",
                        index=False, float_format=f"%.{precision}g")
    _write_meta(bundle, out/f"meta_{tag}.json")

    if not save_png:
        return

    # ---- 6-panel overall map --------------------------------------
    tri=Triangulation(df_raw.x, df_raw.y)
    dx,dy=df_raw.dx,df_raw.dy
    fx,fy=bundle.fit_vec.T
    rx,ry=dx-fx,dy-fy
    data=[[dx,fx,rx],[dy,fy,ry]]
    tit=[["dx raw","dx fit","dx resid"],
         ["dy raw","dy fit","dy resid"]]
    fig,ax=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for i in range(2):
        vmax=max(abs(data[i][0]).max(),abs(data[i][1]).max())
        vres=abs(data[i][2]).max() or 1
        for j in range(3):
            m=ax[i,j].tricontourf(
                tri,data[i][j],levels=100,
                vmin=-(vmax if j<2 else vres),
                vmax= +(vmax if j<2 else vres),
                cmap="coolwarm" if j<2 else "viridis")
            ax[i,j].set_aspect("equal")
            ax[i,j].set_title(tit[i][j])
            ax[i,j].set_xlabel("x [µm]")
            if j==0: ax[i,j].set_ylabel("y [µm]")
            fig.colorbar(m,ax=ax[i,j],shrink=.8)
    fig.suptitle(f"{tag}  RMS = {bundle.rms:.2f} nm")
    fig.savefig(out/f"overall_{tag}.png",dpi=300,bbox_inches="tight")
    plt.close(fig)

    # ---- ヒストグラム（係数） --------------------------------------
    _plot_coeff_hist(bundle, out/f"hist_{tag}.png")

    # ---- 散布図 raw vs fit ----------------------------------------
    _plot_scatter(dx, fx, f"{tag} fit vs raw (dx)",
                  out/f"scatter_{tag}_dx.png")
    _plot_scatter(dy, fy, f"{tag} fit vs raw (dy)",
                  out/f"scatter_{tag}_dy.png")