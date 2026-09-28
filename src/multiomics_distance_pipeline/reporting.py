"""Validation metrics and distribution comparison figures."""

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def distribution_analysis(
    truth, pred, draws, features, scaler, out, modality, plot=True
):
    out.mkdir(parents=True, exist_ok=True)
    if not len(truth):
        return {}
    t = (truth - scaler.mean_) / scaler.scale_
    p = (pred - scaler.mean_) / scaler.scale_
    d = None if draws is None else (draws - scaler.mean_) / scaler.scale_
    lo = hi = None
    if d is not None:
        lo, hi = np.quantile(d, [0.025, 0.975], axis=0)
    rows = []
    for j, f in enumerate(features):
        row = {
            "feature": f,
            "n_masked": len(t),
            "true_mean": truth[:, j].mean(),
            "imputed_mean": pred[:, j].mean(),
            "true_sd": truth[:, j].std(),
            "imputed_sd": pred[:, j].std(),
            "standardized_rmse": float(np.sqrt(np.mean((t[:, j] - p[:, j]) ** 2))),
            "wasserstein_mean": wasserstein_distance(t[:, j], p[:, j]),
            "true_zero_fraction": float(np.mean(truth[:, j] == 0)),
            "imputed_negative_fraction": float(np.mean(pred[:, j] < 0)),
        }
        if d is not None:
            row.update(
                {
                    "predictive_95_coverage": float(
                        np.mean((t[:, j] >= lo[:, j]) & (t[:, j] <= hi[:, j]))
                    ),
                    "predictive_95_width_standardized": float(
                        np.mean(hi[:, j] - lo[:, j])
                    ),
                    "wasserstein_predictive": wasserstein_distance(
                        t[:, j], d[:, :, j].ravel()
                    ),
                }
            )
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / f"{modality}_feature_metrics.csv", index=False)
    if plot:
        # Feature selection uses training scale only, never held-out performance.
        selected = np.argsort(scaler.var_)[-3:]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        axes = axes.ravel()
        for ax, j in zip(axes, [None, *selected]):
            a = t.ravel() if j is None else t[:, j]
            b = p.ravel() if j is None else p[:, j]
            vals = [a, b]
            if d is not None:
                vals.append(d.ravel() if j is None else d[:, :, j].ravel())
            edges = np.histogram_bin_edges(np.concatenate(vals), bins=60)
            for v, label, color in zip(
                vals,
                ["Hidden truth", "Imputed mean", "Predictive draws"],
                ["black", "#2274a5", "#db7c26"],
            ):
                ax.hist(
                    v,
                    bins=edges,
                    density=True,
                    histtype="step",
                    lw=1.5,
                    label=label,
                    color=color,
                )
            ax.set_title(
                "All features pooled" if j is None else features[j], fontsize=8
            )
            ax.set_xlabel("Value standardized by training data")
            ax.set_ylabel("Density")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{modality}: mean versus predictive distribution")
        fig.tight_layout()
        fig.savefig(out / f"{modality}_distributions.png", dpi=160)
        plt.close(fig)
    metrics = {
        f"{modality}_standardized_rmse": float(np.sqrt(np.mean((t - p) ** 2))),
        f"{modality}_wasserstein_mean": float(
            np.mean([r["wasserstein_mean"] for r in rows])
        ),
    }
    if d is not None:
        metrics[f"{modality}_coverage95"] = float(np.mean((t >= lo) & (t <= hi)))
    return metrics


def cross_omics_preservation(true_m, true_t, pred_m, pred_t):
    def cross(m, t):
        x = np.column_stack([m, t])
        sd = x.std(0)
        z = (x - x.mean(0)) / np.where(sd > 0, sd, 1)
        return (z[:, : m.shape[1]].T @ z[:, m.shape[1] :] / len(x)).ravel()

    a = cross(true_m, true_t)
    b = cross(pred_m, pred_t)
    return (
        float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    )


def group_distances(reference, predicted, metadata, out):
    group_col = next(
        (
            c
            for c in [
                "PATGROUPFINAL_C",
                "Study.Group",
                "PatientGroup",
                "PATIENTGROUP",
                "patient_group",
                "Disease",
            ]
            if c in metadata
        ),
        None,
    )
    if group_col is None:
        return
    labels = metadata[group_col].fillna("Unknown").astype(str).to_numpy()
    groups = sorted(set(labels))
    rows = []
    for a in groups:
        for b in groups:
            i = np.flatnonzero(labels == a)
            j = np.flatnonzero(labels == b)
            true = reference[np.ix_(i, j)]
            pred = predicted[np.ix_(i, j)]
            if a == b:
                idx = np.triu_indices(len(i), 1)
                true = true[idx]
                pred = pred[idx]
            if true.size:
                rows.append(
                    {
                        "group_a": a,
                        "group_b": b,
                        "reference_mean_distance": true.mean(),
                        "predicted_mean_distance": pred.mean(),
                    }
                )
    pd.DataFrame(rows).to_csv(out / "patient_group_distances.csv", index=False)


def summarize_validation(long, out):
    metric_cols = [
        c
        for c in long
        if c not in ["repeat", "fold", "method", "missing_rate"]
        and pd.api.types.is_numeric_dtype(long[c])
    ]
    summary = long.groupby(["method", "missing_rate"])[metric_cols].agg(
        ["mean", "std", "count"]
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(out / "validation_summary.csv", index=False)
    paired = long.pivot(
        index=["repeat", "fold", "missing_rate"], columns="method", values="mantel_r"
    )
    paired["gaussian_minus_knn"] = paired["gaussian"] - paired["knn_pca"]
    paired.to_csv(out / "paired_mantel_comparisons.csv")
    for metric, label in [
        ("mantel_r", "Mantel Pearson r (fixed training geometry)"),
        ("official_style_r", "Official-style Mantel r (evaluation-only PCA)"),
        ("prediction_seconds", "Prediction runtime, seconds"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, sub in long.groupby("method"):
            g = sub.groupby("missing_rate")[metric].agg(["mean", "std"]).fillna(0)
            ax.errorbar(
                g.index * 100,
                g["mean"],
                yerr=g["std"],
                marker="o",
                capsize=3,
                label=method,
            )
        ax.set_xlabel("Held-out samples with one block masked (%)")
        ax.set_ylabel(label)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{metric}_by_missingness.png", dpi=170)
        plt.close(fig)
