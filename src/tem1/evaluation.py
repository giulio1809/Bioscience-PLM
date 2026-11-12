"""Rank-based evaluation and diagnostic figures; no fitting or score sign tuning."""

from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr


def evaluate(observed, predicted) -> dict:
    observed, predicted = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    if observed.ndim != 1 or predicted.shape != observed.shape or len(observed) < 3:
        raise ValueError("Evaluation requires at least three paired scalar observations")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Evaluation requires finite scores; no silent row filtering")
    if np.unique(observed).size < 2 or np.unique(predicted).size < 2:
        raise ValueError("Spearman correlation is undefined for constant scores")
    rho = float(spearmanr(observed, predicted).statistic)
    return {
        "n_variants": len(observed), "spearman_rho": rho,
        "score_direction": "higher predicted and experimental scores mean more favorable",
        "ties": "average ranks (scipy.stats.spearmanr)",
    }


def plot_results(observed, predicted, positions, output: Path, title: str, score_label: str = "ESM-2 masked log ratio") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observed, predicted = np.asarray(observed), np.asarray(predicted)
    rho = evaluate(observed, predicted)["spearman_rho"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    axes[0].scatter(observed, predicted, s=7, alpha=0.2, rasterized=True)
    axes[0].set(xlabel="Experimental DMS_score (higher = fitter)", ylabel=score_label)
    axes[1].scatter(rankdata(observed), rankdata(predicted), s=7, alpha=0.2, rasterized=True)
    axes[1].set(xlabel="Experimental rank", ylabel="Predicted rank")
    fig.suptitle(f"{title}\nn = {len(observed):,}; Spearman ρ = {rho:.3f}")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"correlation.{extension}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), layout="constrained")
    unique = sorted(set(positions))
    positions = np.asarray(positions)
    for ax, values, label in zip(axes, (observed, predicted), ("Mean DMS_score", "Mean masked log ratio")):
        ax.plot(unique, [values[positions == p].mean() for p in unique], linewidth=1)
        ax.set(ylabel=label, xlabel="Position in full ProteinGym reference (1-based)")
    fig.suptitle("Per-position means over measured substitutions (coverage may differ)")
    fig.savefig(output / "position_profile.png", dpi=180)
    plt.close(fig)
