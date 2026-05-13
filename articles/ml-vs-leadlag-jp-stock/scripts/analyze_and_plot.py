"""
Generate figures + numeric analysis for the Zenn article:
"機械学習 vs リードラグ戦略：日本株予測の2つのアプローチ"

Inputs (under /Users/shigeru.abe/autoresearch/):
  - live/ml_val_backtest_nav_log.csv       ML val 期間バックテスト
  - live/nav_log.csv                        ML ライブ運用
  - live/daily_ll_{strategy}_nav_log.csv   リードラグ4戦略
  - live/daily_ll_{strategy}_live_nav_log.csv  リードラグ ライブ期間切り出し
  - results.tsv                              autoresearch 進化譜
  - Nikkei225 via yfinance

Outputs (figures/):
  fig02_evolution.png      autoresearch 進化譜
  fig03_ml_val.png         ML val バックテスト 累積リターン
  fig04_ll_full.png        リードラグ4戦略 累積リターン (2015-)
  fig05_battle.png         ML vs PCA_SUB 期間別 3パネル
  fig06_drawdown.png       ドローダウン推移
  fig07_market_wind.png    市場の風 散布図 (回帰付き)
  fig08_rolling_beta.png   rolling 60日 β 推移
  fig09_B_matrix.png       PCA SUB 伝播行列 B ヒートマップ
  fig10_live.png           ライブ運用 (2026-04-01〜)

Usage:
    uv run --with matplotlib --with yfinance --with pandas --with numpy \\
        python3 analyze_and_plot.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

import yfinance as yf

BASE = Path("/Users/shigeru.abe/autoresearch")
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

JA_FONT_CANDIDATES = [
    "Hiragino Sans", "YuGothic", "Yu Gothic", "Hiragino Maru Gothic Pro",
    "Noto Sans CJK JP", "TakaoGothic", "IPAGothic",
]
COLORS = {
    "ML":         "#d62728",
    "ML_live":    "#a3192d",
    "PCA_SUB":    "#2ca02c",
    "PCA_PLAIN":  "#888888",
    "MOM":        "#1f77b4",
    "DOUBLE":     "#9467bd",
    "Nikkei225":  "#cccccc",
}


def setup_japanese_font():
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for cand in JA_FONT_CANDIDATES:
        if cand in avail:
            matplotlib.rcParams["font.family"] = cand
            print(f"[font] using {cand}")
            return
    print("[font] no Japanese font found; using default")


def load_nav(name):
    """Load and normalize NAV log; rebase initial to 100,000."""
    p = BASE / "live" / name
    df = pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    init = float(df["nav"].iloc[0])
    if init != 100_000.0:
        f = 100_000.0 / init
        df["nav"] = df["nav"].astype(float) * f
    return df


def slice_period(df, start, end):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = df[(df["date"] >= s) & (df["date"] <= e)].copy()
    if len(out) > 0:
        init = float(out["nav"].iloc[0])
        out["nav"] = out["nav"].astype(float) * (100_000.0 / init)
        out["cumulative_return"] = out["nav"] / 100_000.0 - 1.0
    return out.reset_index(drop=True)


def metrics(df):
    if len(df) < 2:
        return {}
    r = df["daily_return"].astype(float).values
    nav = df["nav"].astype(float).values
    AR = float(r.mean() * 252 * 100)
    RISK = float(r.std() * np.sqrt(252) * 100)
    SH = AR / RISK if RISK > 0 else float("nan")
    rm = np.maximum.accumulate(nav)
    MDD = float(abs(((nav - rm) / rm).min()) * 100)
    cum = float((nav[-1] / nav[0] - 1) * 100)
    return {"AR": AR, "RISK": RISK, "Sharpe": SH, "MDD": MDD, "cum": cum,
            "final_nav": float(nav[-1]), "days": len(df)}


def fig02_evolution():
    df = pd.read_csv(BASE / "results.tsv", sep="\t",
                     engine="python", on_bad_lines="skip")
    df = df.dropna(subset=["val_sharpe"])
    df["val_sharpe"] = pd.to_numeric(df["val_sharpe"], errors="coerce")
    df = df.dropna(subset=["val_sharpe"]).reset_index(drop=True)
    df["i"] = df.index

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    keep = df[df["status"] == "keep"]
    disc = df[df["status"] == "discard"]
    crash = df[df["status"] == "crash"]

    ax.scatter(disc["i"], disc["val_sharpe"], s=10, c="#cccccc",
               label=f"discard ({len(disc)})", alpha=0.6)
    ax.scatter(crash["i"], crash["val_sharpe"], s=10, c="#000000",
               label=f"crash ({len(crash)})", marker="x", alpha=0.5)
    ax.scatter(keep["i"], keep["val_sharpe"], s=80, c="#d62728",
               label=f"keep ({len(keep)})", edgecolors="black", linewidths=0.6, zorder=5)

    # Running best
    running_best = []
    cur = -np.inf
    for v in df["val_sharpe"]:
        cur = max(cur, v)
        running_best.append(cur)
    ax.plot(df["i"], running_best, color="#d62728", lw=1.6, alpha=0.9, label="running best")

    # Milestones — pick first occurrence of each landmark in keeps
    milestones = [
        (0,    "ベースライン\nRidge α=1"),
        (None, "LightGBM 単体\n(Sharpe 1.0+)"),
        (None, "LGB+Ridge ensemble"),
        (None, "Ridge α tuning\n(Sharpe 1.94)"),
        (None, "Leverage + REB tuning"),
    ]
    keep_idx = keep["i"].tolist()
    sharpe_vals = keep["val_sharpe"].tolist()
    if len(keep_idx) >= 5:
        picks = [keep_idx[0], keep_idx[1], keep_idx[3], keep_idx[7], keep_idx[-1]]
        labels = ["ベースライン\nRidge α=1", "LightGBM 単体", "LGB+Ridge ensemble",
                  "Ridge α tuning", "Leverage + REB\ntuning"]
        for x, lbl in zip(picks, labels):
            y = df.loc[df["i"] == x, "val_sharpe"].iloc[0]
            ax.annotate(lbl, xy=(x, y), xytext=(x + 30, y + 0.35),
                        fontsize=8.5, ha="left",
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    ax.set_xlabel("実験番号 (時系列)")
    ax.set_ylabel("val_sharpe")
    ax.set_title("autoresearch 進化譜 — LLM エージェントが train.py を改造して Sharpe を最大化")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(min(-1.0, df["val_sharpe"].min() * 1.1),
                df["val_sharpe"].max() * 1.1)
    plt.tight_layout()
    out = FIG_DIR / "fig02_evolution.png"
    plt.savefig(out)
    plt.close()
    print(f"  → {out}")


def fig03_ml_val(nikkei):
    df = load_nav("ml_val_backtest_nav_log.csv")
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    cum = (df["nav"].astype(float).values / 100_000.0 - 1) * 100
    ax.plot(df["date"], cum, color=COLORS["ML"], lw=2.0, label="ML (LGB+Ridge ensemble)")
    # Nikkei aligned
    nk = nikkei[(nikkei.index >= df["date"].iloc[0]) & (nikkei.index <= df["date"].iloc[-1])]
    nk_cum = (nk["Close"].values / nk["Close"].iloc[0] - 1) * 100
    ax.plot(nk.index, nk_cum, color=COLORS["Nikkei225"], lw=1.2, label="日経225 (^N225)")
    ax.axhline(0, color="black", lw=0.5)
    m = metrics(df)
    ax.set_xlabel("日付")
    ax.set_ylabel("累積リターン (%)")
    ax.set_title(f"機械学習バックテスト (val期間 2024-2026, 570営業日)\n"
                 f"累積 {m['cum']:+.1f}%  Sharpe {m['Sharpe']:.2f}  MaxDD {m['MDD']:.1f}%")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "fig03_ml_val.png"
    plt.savefig(out)
    plt.close()
    print(f"  → {out}")


def fig04_ll_full():
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    for s, color in [("pca_sub",   COLORS["PCA_SUB"]),
                      ("double",    COLORS["DOUBLE"]),
                      ("mom",       COLORS["MOM"]),
                      ("pca_plain", COLORS["PCA_PLAIN"])]:
        df = load_nav(f"daily_ll_{s}_nav_log.csv")
        cum = (df["nav"].astype(float).values / 100_000.0 - 1) * 100
        lw = 2.2 if s == "pca_sub" else 1.4
        label = {"pca_sub": "PCA_SUB (提案手法)",
                 "pca_plain": "PCA_PLAIN (正則化なし)",
                 "mom": "MOM (モメンタム)",
                 "double": "DOUBLE (MOM×PCA SUB)"}[s]
        ax.plot(df["date"], cum, color=color, lw=lw, label=label)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("日付")
    ax.set_ylabel("累積リターン (%)")
    ax.set_title("リードラグ4戦略の累積リターン (2015-01-05 → 2026-05-12)\n"
                 "Nakagawa et al. 2026 再現実装、日次リバランス、片道2bps")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "fig04_ll_full.png"
    plt.savefig(out)
    plt.close()
    print(f"  → {out}")


def fig05_battle():
    """3 panels: 全期間 / val期間 / ライブ"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    ml_val = load_nav("ml_val_backtest_nav_log.csv")
    ml_live = load_nav("nav_log.csv")
    sub = load_nav("daily_ll_pca_sub_nav_log.csv")

    panels = [
        ("全期間 (2015-2026)\nリードラグ独擅場",
         ("2015-01-01", "2026-05-13"), ax := axes[0], None),
        ("ML val 期間 (2024-2026)\nML 躍進",
         ("2024-01-01", "2026-05-13"), axes[1], ml_val),
        ("直近ライブ (2026-04-)\nML 圧勝",
         ("2026-04-01", "2026-05-13"), axes[2], ml_live),
    ]
    for title, (s, e), ax, ml in panels:
        sub_slice = slice_period(sub, s, e)
        if len(sub_slice) > 0:
            cum = (sub_slice["nav"] / 100_000.0 - 1) * 100
            ax.plot(sub_slice["date"], cum, color=COLORS["PCA_SUB"], lw=1.8,
                    label="リードラグ (PCA_SUB)")
        if ml is not None and len(ml) > 0:
            ml_slice = slice_period(ml, s, e)
            cum = (ml_slice["nav"] / 100_000.0 - 1) * 100
            ax.plot(ml_slice["date"], cum, color=COLORS["ML"], lw=1.8,
                    label="機械学習 (ML)")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("累積リターン (%)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
    plt.suptitle("機械学習 vs リードラグ：時代で勝者が入れ替わる",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "fig05_battle.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def fig06_drawdown():
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    for path, color, label in [
        ("ml_val_backtest_nav_log.csv", COLORS["ML"], "機械学習 (ML)"),
        ("daily_ll_pca_sub_nav_log.csv", COLORS["PCA_SUB"], "リードラグ (PCA_SUB)"),
    ]:
        df = load_nav(path)
        # restrict to ML val period for fair compare
        df = slice_period(df, "2024-01-01", "2026-05-13")
        if len(df) == 0:
            continue
        nav = df["nav"].astype(float).values
        rm = np.maximum.accumulate(nav)
        dd = (nav - rm) / rm * 100
        ax.fill_between(df["date"], dd, 0, color=color, alpha=0.3)
        ax.plot(df["date"], dd, color=color, lw=1.2, label=label)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("日付")
    ax.set_ylabel("ドローダウン (%)")
    ax.set_title("ドローダウン推移 (ML val 期間 2024-2026)")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "fig06_drawdown.png"
    plt.savefig(out)
    plt.close()
    print(f"  → {out}")


def fig07_market_wind(nikkei):
    """市場の風 散布図 (Nikkei vs 各戦略 daily return)"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)
    n_ret = nikkei["Close"].pct_change().dropna()

    for ax, (path, color, label) in zip(axes, [
        ("ml_val_backtest_nav_log.csv", COLORS["ML"], "機械学習 (ML)"),
        ("daily_ll_pca_sub_nav_log.csv", COLORS["PCA_SUB"], "リードラグ (PCA_SUB)"),
    ]):
        df = load_nav(path)
        # align dates between strategy and Nikkei (val period for fair compare)
        df = slice_period(df, "2024-01-01", "2026-05-13")
        df = df.set_index("date")
        s_ret = df["daily_return"].astype(float)
        joined = pd.concat([n_ret.rename("nikkei"), s_ret.rename("strat")],
                           axis=1).dropna()
        x = joined["nikkei"].values * 100
        y = joined["strat"].values * 100
        # OLS regression
        beta, alpha = np.polyfit(x, y, 1)
        # R^2
        y_pred = beta * x + alpha
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        ax.scatter(x, y, s=12, alpha=0.5, color=color)
        xx = np.linspace(x.min(), x.max(), 50)
        ax.plot(xx, beta * xx + alpha, color="black", lw=1.5, alpha=0.8,
                label=f"β={beta:+.3f}, α={alpha:+.3f}%, R²={r2:.3f}")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel("日経225 日次リターン (%)")
        ax.set_ylabel(f"{label} 日次リターン (%)")
        ax.set_title(label, fontsize=11)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("市場の風 vs 戦略リターン — ML は風に乗り、リードラグは風と無関係",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "fig07_market_wind.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def fig08_rolling_beta(nikkei):
    """rolling 60-day beta to Nikkei"""
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    n_ret = nikkei["Close"].pct_change().dropna()
    WINDOW = 60

    for path, color, label in [
        ("ml_val_backtest_nav_log.csv", COLORS["ML"], "機械学習 (ML)"),
        ("daily_ll_pca_sub_nav_log.csv", COLORS["PCA_SUB"], "リードラグ (PCA_SUB)"),
    ]:
        df = load_nav(path)
        df = slice_period(df, "2024-01-01", "2026-05-13")
        df = df.set_index("date")
        s_ret = df["daily_return"].astype(float)
        joined = pd.concat([n_ret.rename("n"), s_ret.rename("s")], axis=1).dropna()
        betas, dates = [], []
        for i in range(WINDOW, len(joined)):
            sub = joined.iloc[i - WINDOW:i]
            xb = sub["n"].values
            yb = sub["s"].values
            if xb.std() < 1e-9:
                betas.append(np.nan)
            else:
                b = np.cov(xb, yb)[0, 1] / np.var(xb)
                betas.append(b)
            dates.append(joined.index[i])
        ax.plot(dates, betas, color=color, lw=1.5, label=label)
    ax.axhline(0, color="black", lw=0.6, linestyle="--", alpha=0.6)
    ax.set_xlabel("日付")
    ax.set_ylabel("rolling 60日 β (vs 日経225)")
    ax.set_title("市場の風への感応度 (rolling 60日 β) — "
                 "ML は変動的にレバレッジ、リードラグは β ≈ 0 で安定")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "fig08_rolling_beta.png"
    plt.savefig(out)
    plt.close()
    print(f"  → {out}")


def fig09_B_matrix():
    """PCA SUB の伝播行列 B = V_J V_U^T を再計算して可視化"""
    sys.path.insert(0, str(BASE))
    from lead_lag_daily_train import (
        US_TICKERS, JP_TICKERS, N_US, N_JP, L, K, LAMBDA,
        build_prior_directions, compute_cfull, build_C0,
        download_universe, CFG,
    )

    us_close, jp_open, jp_close = download_universe(
        CFG["CFULL_START"], CFG["CFULL_END"],
    )
    us_cc = us_close.pct_change().dropna()
    jp_cc_full = jp_close.pct_change().dropna()
    common = sorted(set(us_cc.index) & set(jp_cc_full.index))
    V0 = build_prior_directions()
    C_full = compute_cfull(us_cc, jp_cc_full, common)
    C0 = build_C0(V0, C_full)

    # Take a representative window (mid-period) for B computation
    joint = pd.concat([us_cc, jp_cc_full], axis=1, join="inner").dropna()
    joint.columns = US_TICKERS + JP_TICKERS
    mid = len(joint) // 2
    win = joint.iloc[mid - L:mid].values
    mu, sd = win.mean(0), win.std(0) + 1e-12
    win_z = (win - mu) / sd
    C_t = np.corrcoef(win_z, rowvar=False)
    C_reg = (1 - LAMBDA) * C_t + LAMBDA * C0
    evals, evecs = np.linalg.eigh(C_reg)
    order = np.argsort(evals)[::-1]
    V = evecs[:, order[:K]]
    V_U = V[:N_US, :]
    V_J = V[N_US:, :]
    B = V_J @ V_U.T

    us_names = [s["name"] for s in CFG["US_SECTORS"]]
    jp_names = [s["name"] for s in CFG["JP_SECTORS"]]

    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
    vmax = max(abs(B.min()), abs(B.max()))
    im = ax.imshow(B, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(N_US))
    ax.set_xticklabels(us_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(N_JP))
    ax.set_yticklabels(jp_names, fontsize=9)
    ax.set_xlabel("米国セクター ETF (SPDR)")
    ax.set_ylabel("日本セクター ETF (TOPIX-17)")
    ax.set_title("PCA_SUB の伝播行列 B = V_J V_U^T (上位3固有空間、λ=0.9)\n"
                 "赤=正の伝播、青=負の伝播")
    fig.colorbar(im, ax=ax, label="伝播係数")
    plt.tight_layout()
    out = FIG_DIR / "fig09_B_matrix.png"
    plt.savefig(out)
    plt.close()
    print(f"  → {out}")


def fig10_live():
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    ml = load_nav("nav_log.csv")
    cum = (ml["nav"] / 100_000.0 - 1) * 100
    ax.plot(ml["date"], cum, color=COLORS["ML"], lw=2.4, label="機械学習 (ML)",
            marker="o", markersize=4)
    for s, color, label in [
        ("pca_sub",   COLORS["PCA_SUB"],   "リードラグ (PCA_SUB)"),
        ("double",    COLORS["DOUBLE"],    "DOUBLE"),
        ("mom",       COLORS["MOM"],       "MOM"),
        ("pca_plain", COLORS["PCA_PLAIN"], "PCA_PLAIN"),
    ]:
        df = load_nav(f"daily_ll_{s}_live_nav_log.csv")
        if len(df) == 0:
            continue
        cum = (df["nav"] / 100_000.0 - 1) * 100
        ax.plot(df["date"], cum, color=color, lw=1.4 if s != "pca_sub" else 2.0,
                label=label, marker=".", markersize=3)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("日付")
    ax.set_ylabel("累積リターン (%)")
    ax.set_title("ライブ運用 (2026-04-01 → 2026-05-12, 27営業日)\n"
                 "ML が一方的に勝った直近1ヶ月")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "fig10_live.png"
    plt.savefig(out)
    plt.close()
    print(f"  → {out}")


def print_regression_summary(nikkei):
    """Print α / β / R² for ML and PCA_SUB on the val period."""
    n_ret = nikkei["Close"].pct_change().dropna()
    print("\n=== 市場の風 回帰分析 (val 期間 2024-2026) ===")
    print(f"{'Strategy':<22} | {'β':>8} | {'α (日次%)':>10} | {'α (年率%)':>10} | {'R²':>6}")
    print("-" * 70)
    for path, label in [
        ("ml_val_backtest_nav_log.csv", "ML (LGB+Ridge)"),
        ("daily_ll_pca_sub_nav_log.csv", "PCA_SUB (リードラグ)"),
        ("daily_ll_mom_nav_log.csv", "MOM"),
        ("daily_ll_pca_plain_nav_log.csv", "PCA_PLAIN"),
        ("daily_ll_double_nav_log.csv", "DOUBLE"),
    ]:
        df = load_nav(path)
        df = slice_period(df, "2024-01-01", "2026-05-13")
        s_ret = df.set_index("date")["daily_return"].astype(float)
        j = pd.concat([n_ret.rename("n"), s_ret.rename("s")], axis=1).dropna()
        x, y = j["n"].values, j["s"].values
        beta, alpha = np.polyfit(x, y, 1)
        yp = beta * x + alpha
        r2 = 1 - ((y - yp) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"{label:<22} | {beta:+8.4f} | {alpha*100:+10.4f} | "
              f"{alpha*252*100:+10.2f} | {r2:6.3f}")


def main():
    setup_japanese_font()
    print("Downloading Nikkei225 ...")
    n225 = yf.download("^N225", start="2015-01-01", end="2026-05-13",
                       auto_adjust=True, progress=False)
    # flatten possible MultiIndex columns
    if isinstance(n225.columns, pd.MultiIndex):
        n225.columns = n225.columns.get_level_values(0)
    print(f"  Nikkei225: {len(n225)} days")
    print()
    print("Generating figures ...")
    fig02_evolution()
    fig03_ml_val(n225)
    fig04_ll_full()
    fig05_battle()
    fig06_drawdown()
    fig07_market_wind(n225)
    fig08_rolling_beta(n225)
    fig09_B_matrix()
    fig10_live()
    print_regression_summary(n225)


if __name__ == "__main__":
    main()
