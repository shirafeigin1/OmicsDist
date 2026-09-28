"""Cross-validation with preprocessing fitted on training observations."""

import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from .bayesian_gibbs_pipeline import FittedModels, METHODS
from .distances import DistanceSpace, mantel
from .reporting import (
    distribution_analysis,
    cross_omics_preservation,
    group_distances,
    summarize_validation,
)


def run_validation_experiment(data, output_dir, cfg, resume=False):
    out = output_dir / "validation"
    out.mkdir(parents=True, exist_ok=True)
    complete = np.flatnonzero((data.micro_available & data.metab_available).to_numpy())
    if len(complete) // cfg.folds < 5:
        raise ValueError("Too few complete samples per validation fold.")
    splitter = RepeatedKFold(
        n_splits=cfg.folds, n_repeats=cfg.repeats, random_state=cfg.seed
    )
    rows = []
    for index, (train_local, test_local) in enumerate(splitter.split(complete)):
        rep, fold = divmod(index, cfg.folds)
        rep += 1
        fold += 1
        fold_out = out / f"repeat_{rep:02d}_fold_{fold:02d}"
        fold_out.mkdir(exist_ok=True)
        if resume and (fold_out / "DONE.json").exists():
            rows.extend(pd.read_csv(fold_out / "metrics.csv").to_dict("records"))
            continue
        train, test = complete[train_local], complete[test_local]
        ids = np.asarray(data.sample_ids)
        pd.DataFrame(
            {
                "SampleID": ids[np.r_[train, test]],
                "partition": ["train"] * len(train) + ["test"] * len(test),
            }
        ).to_csv(fold_out / "split.csv", index=False)
        print(f"[CV] repeat {rep}/{cfg.repeats}, fold {fold}/{cfg.folds}", flush=True)
        model = FittedModels().fit(
            data.micro[train],
            data.metab[train],
            cfg,
            fold_out / "fit",
            cfg.seed + index * 10000,
        )
        true_m, true_t = data.micro[test], data.metab[test]
        reference = model.space.distance(true_m, true_t)
        official_reference = (
            DistanceSpace().fit(true_m, true_t).distance(true_m, true_t)
        )
        rng = np.random.default_rng(cfg.seed + index * 10000 + 100)
        order = rng.permutation(len(test))
        fold_rows = []
        for rate in cfg.missing_rates:
            rate_out = fold_out / f"missing_{rate:.3f}"
            rate_out.mkdir(exist_ok=True)
            n = max(2, int(round(len(test) * rate)))
            selected = order[:n]
            mi, ti = np.sort(selected[::2]), np.sort(
                selected[1::2]
            )  # Nested, disjoint masking across rates.
            micro, metab = true_m.copy(), true_t.copy()
            micro[mi] = np.nan
            metab[ti] = np.nan
            mask = pd.DataFrame({"SampleID": ids[test], "masked_block": "none"})
            mask.loc[mi, "masked_block"] = "microbiome"
            mask.loc[ti, "masked_block"] = "metabolome"
            mask.to_csv(rate_out / "mask.csv", index=False)
            for method in METHODS:
                dest = rate_out / method
                dest.mkdir(exist_ok=True)
                print(f"  predicting {rate:.0%} / {method}", flush=True)
                t = time.perf_counter()
                prediction = model.predict(
                    micro, metab, method, cfg.seed + index * 10000 + 200
                )
                distance = model.space.distance(prediction.micro, prediction.metab)
                official_distance = (
                    DistanceSpace()
                    .fit(prediction.micro, prediction.metab)
                    .distance(prediction.micro, prediction.metab)
                )
                pipeline_prediction_time = time.perf_counter() - t
                r, rs, p = mantel(
                    reference, distance, cfg.permutations, cfg.seed + index
                )
                official_r, _, _ = mantel(official_reference, official_distance, 0)
                tri = np.triu_indices(len(test), 1)
                affected = np.zeros(len(test), bool)
                affected[selected] = True
                valid_pairs = affected[tri[0]] | affected[tri[1]]
                a, b = reference[tri][valid_pairs], distance[tri][valid_pairs]
                affected_r = (
                    float(np.corrcoef(a, b)[0, 1])
                    if a.std() > 0 and b.std() > 0
                    else np.nan
                )
                row = {
                    "repeat": rep,
                    "fold": fold,
                    "method": method,
                    "missing_rate": rate,
                    "n_train": len(train),
                    "n_test": len(test),
                    "actual_missing_rate": len(selected) / len(test),
                    "n_mask_micro": len(mi),
                    "n_mask_metab": len(ti),
                    "mantel_r": r,
                    "mantel_spearman": rs,
                    "mantel_p_two_sided": p,
                    "official_style_r": official_r,
                    "affected_pairs_r": affected_r,
                    "cross_omics_correlation_preservation": cross_omics_preservation(
                        true_m, true_t, prediction.micro, prediction.metab
                    ),
                    "fit_seconds": model.fit_times[method]
                    + model.preprocessing_seconds,
                    "prediction_seconds": pipeline_prediction_time,
                    "converged": model.converged(method),
                }
                plot = rate == min(cfg.missing_rates, key=lambda v: abs(v - 0.3))
                row.update(
                    distribution_analysis(
                        true_m[mi],
                        prediction.micro[mi],
                        prediction.micro_draws,
                        data.micro_features,
                        model.micro_pca.scaler,
                        dest,
                        "microbiome",
                        plot,
                    )
                )
                row.update(
                    distribution_analysis(
                        true_t[ti],
                        prediction.metab[ti],
                        prediction.metab_draws,
                        data.metab_features,
                        model.metab_pca.scaler,
                        dest,
                        "metabolome",
                        plot,
                    )
                )
                group_distances(
                    reference,
                    distance,
                    data.metadata.iloc[test].reset_index(drop=True),
                    dest,
                )
                fold_rows.append(row)
        pd.DataFrame(fold_rows).to_csv(fold_out / "metrics.csv", index=False)
        (fold_out / "DONE.json").write_text(json.dumps({"repeat": rep, "fold": fold}))
        rows.extend(fold_rows)
        pd.DataFrame(rows).to_csv(out / "validation_long.csv", index=False)
        del model
    result = pd.DataFrame(rows)
    result.to_csv(out / "validation_long.csv", index=False)
    summarize_validation(result, out)
    return result
