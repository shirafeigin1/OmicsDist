import json
import time
import numpy as np
import pandas as pd
from .bayesian_gibbs_pipeline import FittedModels, METHODS
from .distances import DistanceSpace
from .utils import save_distance_matrix


def run_final_matrices(
    data, output_dir, cfg, resume=False, dataset="dataset", student="shira"
):
    out = output_dir / "final"
    out.mkdir(parents=True, exist_ok=True)
    if resume and (out / "DONE.json").exists():
        return pd.read_csv(out / "runtime_summary.csv")
    complete = (data.micro_available & data.metab_available).to_numpy()
    pd.DataFrame({"SampleID": data.sample_ids, "training_pair": complete}).to_csv(
        out / "training_pairs.csv", index=False
    )
    model = FittedModels().fit(
        data.micro[complete], data.metab[complete], cfg, out / "fit", cfg.seed + 1000000
    )
    rows = []
    for method in METHODS:
        dest = out / method
        dest.mkdir(exist_ok=True)
        print(f"[final] {method}", flush=True)
        t = time.perf_counter()
        pred = model.predict(data.micro, data.metab, method, cfg.seed + 2000000)
        # Fit the final PC1/PC2 distance space after feature-level averaging.
        space = DistanceSpace().fit(pred.micro, pred.metab)
        coords = space.coordinates(pred.micro, pred.metab)
        matrix = space.distance(pred.micro, pred.metab)
        filename = (
            f"{student}_res_{dataset}.csv"
            if method == "gaussian"
            else "knn_pca_distance_matrix.csv"
        )
        save_distance_matrix(matrix, data.sample_ids, out / filename)
        pd.DataFrame(coords, columns=["PC1", "PC2"]).assign(
            SampleID=data.sample_ids
        ).to_csv(dest / "coordinates.csv", index=False)
        missing_m = ~data.micro_available.to_numpy()
        rows.append(
            {
                "method": method,
                "fit_seconds": model.fit_times[method] + model.preprocessing_seconds,
                "predict_export_seconds": time.perf_counter() - t,
                "converged": model.converged(method),
                "matrix": filename,
                "imputed_micro_negative_fraction": (
                    float(np.mean(pred.micro[missing_m] < 0)) if missing_m.any() else 0
                ),
                "max_micro_row_sum_error": (
                    float(np.max(np.abs(pred.micro.sum(1) - 1)))
                    if model.composition.scale == "relative"
                    else np.nan
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(out / "runtime_summary.csv", index=False)
    (out / "DONE.json").write_text(
        json.dumps(
            {
                "completed": True,
                "convergence": {m: model.converged(m) for m in METHODS},
            },
            indent=2,
        )
    )
    return result
