import os
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
from scipy.stats import beta as beta_dist, norm
import matplotlib.pyplot as plt
from numpy.random import SeedSequence, default_rng

# 1) Problem setup

def true_m(x: np.ndarray) -> np.ndarray:
    return np.sin(1.0 / (x / 3.0 + 0.1))


@dataclass
class DataConfig:
    n: int
    alpha: float
    beta: float
    sigma2: float = 1.0
    support_len: float = 1.0  # |supp(X)|; Beta(α,β) on [0,1] => 1.


def simulate_dataset(cfg: DataConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    X = beta_dist.rvs(cfg.alpha, cfg.beta, size=cfg.n, random_state=rng)
    eps = norm.rvs(scale=math.sqrt(cfg.sigma2), size=cfg.n, random_state=rng)
    Y = true_m(X) + eps
    return X, Y


# 2) Pilot estimation of θ22 and σ2 using blocked quartic fits

def _block_indices_sorted_by_X(X: np.ndarray, N: int) -> List[np.ndarray]:
    """
    Partition the sample into N equal-sized blocks after sorting by X.
    If n is not divisible by N, the last block may have a few extra points.
    """
    n = X.shape[0]
    order = np.argsort(X, kind='mergesort')  # stable sort
    base = n // N
    rem = n % N
    blocks = []
    start = 0
    for j in range(N):
        size = base + (1 if j < rem else 0)
        idx_sorted = order[start:start + size]
        blocks.append(idx_sorted)
        start += size
    return blocks


def _fit_quartic_and_second_deriv(xb: np.ndarray, yb: np.ndarray, x_eval: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Use numpy.polyfit (degree=4). Requires at least 5 points; our N_max choice guarantees this.
    coeffs = np.polyfit(xb, yb, deg=4)   # [a, b, c, d, e]
    a, b, c, d, e = coeffs
    y_hat = np.polyval(coeffs, x_eval)
    m2 = 12.0 * a * x_eval**2 + 6.0 * b * x_eval + 2.0 * c
    return y_hat, m2


def blocked_quartic_pilots(X: np.ndarray, Y: np.ndarray, N: int) -> Dict[str, float]:
    """
    For a given N (number of blocks), fit quartic in each block
    and compute the pilot estimates θ22_hat(N) and σ2_hat(N).
    Also return RSS(N) for Cp.
    """
    n = X.shape[0]
    assert N >= 1
    assert 5 * N < n, "Need n > 5N to have positive residual d.f."

    blocks = _block_indices_sorted_by_X(X, N)

    m2_sq_sum = 0.0
    rss_sum = 0.0

    for idx in blocks:
        xb = X[idx]
        yb = Y[idx]
        yhat_b, m2_b = _fit_quartic_and_second_deriv(xb, yb, xb)
        rss_sum += float(np.sum((yb - yhat_b)**2))
        m2_sq_sum += float(np.sum(m2_b**2))

    theta22_hat = m2_sq_sum / n
    sigma2_hat = rss_sum / (n - 5 * N)

    return {"theta22_hat": theta22_hat, "sigma2_hat": sigma2_hat, "RSS": rss_sum}


def N_max_rule(n: int) -> int:
    return max(min(n // 20, 5), 1)


def mallows_Cp_selection(X: np.ndarray, Y: np.ndarray) -> Dict:
    """
    Compute Cp(N) for N = 1...N_max and pick argmin.
    Returns dict with:
      - N_candidates, N_opt
      - per-N theta22_hat, sigma2_hat, RSS, Cp
    """
    n = X.shape[0]
    Nmax = N_max_rule(n)
    Ns = list(range(1, Nmax + 1))

    results = []
    pilots_cache = {}
    for N in Ns:
        pilots_cache[N] = blocked_quartic_pilots(X, Y, N)

    RSS_Nmax = pilots_cache[Nmax]["RSS"]
    denom = RSS_Nmax / (n - 5 * Nmax)

    for N in Ns:
        d = pilots_cache[N]
        CpN = d["RSS"] / denom - (n - 10 * N)
        row = {
            "N": N,
            "theta22_hat": d["theta22_hat"],
            "sigma2_hat": d["sigma2_hat"],
            "RSS": d["RSS"],
            "Cp": CpN
        }
        results.append(row)

    df = pd.DataFrame(results).sort_values("N").reset_index(drop=True)
    N_opt = int(df.loc[df["Cp"].idxmin(), "N"])
    return {
        "N_candidates": Ns, "table": df, "N_opt": N_opt
    }


# 3) AMISE bandwidth with quartic (biweight) kernel constant “35”

def h_amise(n: int, sigma2_hat: float, theta22_hat: float, support_len: float = 1.0) -> float:
    """
    Guard against tiny theta22_hat.
    """
    eps = 1e-12
    t = max(theta22_hat, eps)
    inside = (35.0 * sigma2_hat * support_len) / t
    return (n ** (-1.0 / 5.0)) * (inside ** (1.0 / 5.0))


# 4) One-shot analysis for a single dataset (inspect dependence on N)

def analyze_single_dataset(cfg: DataConfig, rng: np.random.Generator, outdir: str) -> Dict:
    X, Y = simulate_dataset(cfg, rng)
    sel = mallows_Cp_selection(X, Y)
    Nopt = sel["N_opt"]
    tab = sel["table"]  # columns: N, theta22_hat, sigma2_hat, RSS, Cp
    tab["h_hat"] = [h_amise(cfg.n, row.sigma2_hat, row.theta22_hat, cfg.support_len) for _, row in tab.iterrows()]
    os.makedirs(outdir, exist_ok=True)

    # Plot 1: h_hat vs N
    plt.figure()
    plt.plot(tab["N"], tab["h_hat"], marker="o")
    plt.xlabel("Block size N")
    plt.ylabel(r"$\hat h_{\mathrm{AMISE}}$")
    plt.title(f"h vs N  (n={cfg.n}, alpha={cfg.alpha}, beta={cfg.beta})")
    plt.tight_layout()
    f1 = os.path.join(outdir, f"h_vs_N_n{cfg.n}_a{cfg.alpha}_b{cfg.beta}.png")
    plt.savefig(f1, dpi=150)
    plt.close()

    # Plot 2: theta22_hat vs N
    plt.figure()
    plt.plot(tab["N"], tab["theta22_hat"], marker="o")
    plt.xlabel("Block size N")
    plt.ylabel(r"$\widehat{\theta}_{22}(N)$")
    plt.title(f"theta22 vs N  (n={cfg.n}, alpha={cfg.alpha}, beta={cfg.beta})")
    plt.tight_layout()
    f2 = os.path.join(outdir, f"theta22_vs_N_n{cfg.n}_a{cfg.alpha}_b{cfg.beta}.png")
    plt.savefig(f2, dpi=150)
    plt.close()

    # Plot 3: sigma2_hat vs N
    plt.figure()
    plt.plot(tab["N"], tab["sigma2_hat"], marker="o")
    plt.xlabel("Block size N")
    plt.ylabel(r"$\widehat{\sigma}^2(N)$")
    plt.title(f"sigma^2 vs N  (n={cfg.n}, alpha={cfg.alpha}, beta={cfg.beta})")
    plt.tight_layout()
    f3 = os.path.join(outdir, f"sigma2_vs_N_n{cfg.n}_a{cfg.alpha}_b{cfg.beta}.png")
    plt.savefig(f3, dpi=150)
    plt.close()

    return {
        "X": X, "Y": Y, "table": tab, "N_opt": Nopt, "figs": [f1, f2, f3]
    }


# 5) Full simulation across (n, α, β), replicated

@dataclass
class SimGrid:
    n_list: List[int]
    alpha_beta_list: List[Tuple[float, float]]
    R: int = 200  # number of Monte Carlo replicates


def run_simulation(grid: SimGrid, sigma2: float, outdir: str, seed: int = 517) -> pd.DataFrame:
    """
    For each (n, alpha, beta) in the grid, run R replicates:
      - choose N via Mallows' Cp
      - estimate theta22_hat(N_opt), sigma2_hat(N_opt)
      - compute h_hat
    Also produce diagnostic plots of h_hat/theta22_hat/sigma2_hat vs N
    for each (alpha,beta) using independent RNG streams (order-invariant and reproducible).
    """
    rng = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)

    rows = []
    for n in grid.n_list:
        for (a, b) in grid.alpha_beta_list:
            for r in range(grid.R):
                cfg = DataConfig(n=n, alpha=a, beta=b, sigma2=sigma2, support_len=1.0)
                X, Y = simulate_dataset(cfg, rng)
                sel = mallows_Cp_selection(X, Y)
                tab = sel["table"]
                Nopt = sel["N_opt"]
                # grab the Nopt row
                row_opt = tab.loc[tab["N"] == Nopt].iloc[0]
                hhat = h_amise(n, row_opt.sigma2_hat, row_opt.theta22_hat, cfg.support_len)

                rows.append({
                    "n": n, "alpha": a, "beta": b, "rep": r + 1,
                    "N_opt": Nopt,
                    "theta22_hat": row_opt.theta22_hat,
                    "sigma2_hat": row_opt.sigma2_hat,
                    "h_hat": hhat
                })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "simulation_results.csv")
    df.to_csv(csv_path, index=False)

    # For each fixed (alpha, beta) dataset, show one replicate’s h_hat/theta22_hat/sigma2_hat vs N
    # Use the same representative n0 (second smallest if possible) for comparability
    n_sorted = sorted(grid.n_list)
    n0 = n_sorted[min(1, len(n_sorted) - 1)]

    # Spawn an independent child RNG for each (alpha,beta) so diagnostics are invariant to call order
    diag_base_ss = SeedSequence(seed)
    diag_children = diag_base_ss.spawn(len(grid.alpha_beta_list))

    for (a0, b0), ss in zip(grid.alpha_beta_list, diag_children):
        try:
            rng_diag = default_rng(ss)  # fresh, independent RNG stream for this scenario
            cfg0 = DataConfig(n=n0, alpha=a0, beta=b0, sigma2=sigma2)
            _ = analyze_single_dataset(cfg0, rng_diag, outdir=outdir)
        except Exception as e:
            print(f"Single-scenario diagnostic plots failed for (alpha={a0}, beta={b0}):", e)

    # For each (alpha,beta), show h_hat vs n (boxplots) and N_opt vs n (barplots)
    for (a, b) in grid.alpha_beta_list:
        sub = df[(df["alpha"] == a) & (df["beta"] == b)].copy()
        if sub.empty:
            continue

        n_vals = sorted(grid.n_list)

        # h_hat vs n (boxplots)
        fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(n_vals)), 4.6), constrained_layout=False)
        data = [sub[sub["n"] == n]["h_hat"].values for n in n_vals]
        ax.boxplot(data, labels=[str(n) for n in n_vals], showfliers=False)
        ax.set_xlabel("n", labelpad=6)
        ax.set_ylabel(r"$\hat h_{\mathrm{AMISE}}$", labelpad=6)
        ax.set_title(rf"$\hat h$ vs $n$  (alpha={a}, beta={b})", pad=10)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(0)
            lbl.set_horizontalalignment("center")
        fig.tight_layout(pad=1.2)
        fname1 = os.path.join(outdir, f"h_vs_n_alpha{a}_beta{b}.png")
        fig.savefig(fname1, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # N_opt vs n (barplots)
        N_vals = sorted(sub["N_opt"].unique().astype(int))
        counts = np.zeros((len(n_vals), len(N_vals)), dtype=int)
        for i, ncat in enumerate(n_vals):
            slice_n = sub[sub["n"] == ncat]["N_opt"].astype(int)
            for j, Ncat in enumerate(N_vals):
                counts[i, j] = (slice_n == Ncat).sum()
        x = np.arange(len(n_vals))       
        k = len(N_vals)                  
        total_width = 0.8
        bar_w = total_width / max(k, 1)
        offsets = x[:, None] + (np.arange(k) - (k - 1) / 2) * bar_w
        fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(n_vals)), 4.8), constrained_layout=False)
        annotate_counts = (len(n_vals) <= 8) and (len(N_vals) <= 6)
        for j, Ncat in enumerate(N_vals):
            bars = ax.bar(offsets[:, j], counts[:, j], width=bar_w, edgecolor="black",
                          label=f"$N_{{\\mathrm{{opt}}}}={Ncat}$")
            if annotate_counts:
                for rect, c in zip(bars, counts[:, j]):
                    if c > 0:
                        ax.text(rect.get_x() + rect.get_width() / 2,
                                rect.get_height() + max(0.5, 0.02 * counts.max()),
                                str(c),
                                ha="center", va="bottom", fontsize=8, clip_on=True)
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in n_vals])
        if len(n_vals) > 6:
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(20)
                lbl.set_horizontalalignment("right")
        ax.set_xlabel("n", labelpad=6)
        ax.set_ylabel("Count", labelpad=6)
        ax.set_title(f"$N_{{opt}}$ vs n  (alpha={a}, beta={b})", pad=10)
        fig.subplots_adjust(right=0.8, bottom=0.12, top=0.9)
        ax.legend(title=r"$N_{\mathrm{opt}}$", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, frameon=False)
        ymax = counts.max() if counts.size else 1
        ax.set_ylim(0, ymax * 1.15 + 0.5)
        fname2 = os.path.join(outdir, f"Nopt_vs_n_alpha{a}_beta{b}.png")
        fig.savefig(fname2, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return df


# 6) Main

if __name__ == "__main__":
    outdir = "results"
    grid = SimGrid(
        n_list =[100, 1000, 10000, 100000],
        alpha_beta_list=[(2.0, 2.0), (2.0, 5.0), (5.0, 2.0), (0.5, 0.5)],
        R=100
    )
    sigma2 = 1.0
    df = run_simulation(grid, sigma2=sigma2, outdir=outdir, seed=517)

    summary = (
        df.groupby(["alpha", "beta", "n"])
          .agg(h_median=("h_hat", "median"),
               h_iqr=("h_hat", lambda x: np.percentile(x, 75) - np.percentile(x, 25)),
               Nopt_mode=("N_opt", lambda s: s.mode().iat[0] if not s.mode().empty else np.nan))
          .reset_index()
    )
    print(summary.to_string(index=False))
