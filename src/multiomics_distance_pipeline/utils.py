from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ID_COL = "SampleID"
ALT_ID_COL = "Unnamed: 0"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_sample_id(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a standardized SampleID column."""
    df = df.copy()
    if ID_COL not in df.columns:
        if ALT_ID_COL in df.columns:
            df = df.rename(columns={ALT_ID_COL: ID_COL})
        else:
            df = df.rename(columns={df.columns[0]: ID_COL})
    df[ID_COL] = df[ID_COL].astype(str)
    return df


def upper_triangle_values(distance_matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(distance_matrix.shape[0], k=1)
    return distance_matrix[idx]


def mantel_correlation(
    reference: np.ndarray, predicted: np.ndarray, method: str = "pearson"
) -> float:
    """Mantel-style correlation between upper triangles of two distance matrices.

    This compares distance structure rather than raw distance scale.
    """
    ref = upper_triangle_values(np.asarray(reference, dtype=float))
    pred = upper_triangle_values(np.asarray(predicted, dtype=float))
    mask = np.isfinite(ref) & np.isfinite(pred)
    ref = ref[mask]
    pred = pred[mask]
    if len(ref) < 3 or np.std(ref) == 0 or np.std(pred) == 0:
        return float("nan")
    if method == "spearman":
        return float(spearmanr(ref, pred).correlation)
    return float(pearsonr(ref, pred).statistic)


def save_distance_matrix(
    distance_matrix: np.ndarray, sample_ids: Iterable[str], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ids = list(sample_ids)
    pd.DataFrame(distance_matrix, index=ids, columns=ids).to_csv(output_path)
