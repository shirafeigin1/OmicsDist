# OmicsDist

Multi-omics imputation and sample-distance estimation using Gaussian Gibbs regression and PCA-based KNN.

## Input structure

Put each dataset in its own directory, `data/name/`, containing:

- `metadata.csv`
- `microbiome.csv`
- `metabolome.csv`

Replace `name` in the commands below with the exact directory name. Dataset names are not predefined. Quote names that contain spaces.

The first column identifies each sample. Tables are aligned by identifier, and matrix order follows metadata. Omics columns must be numeric. Missing values must form entire omics blocks, and each sample must have at least one observed modality. Microbiome inputs must be relative abundances or centered CLR values. Disease groups in `PATGROUPFINAL_C` or `Study.Group` are used only for reporting.

## Windows PowerShell

Open PowerShell in the extracted `OmicsDist` directory. Create the environment once, using Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run a dataset:

```powershell
.\.venv\Scripts\python.exe .\src\run_pipeline.py --dataset name
```

## Linux

Use Python 3.10 or newer. Create the environment once:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Run a dataset:

```bash
.venv/bin/python src/run_pipeline.py --dataset name
```

## Pipeline

Each run performs five-fold cross-validation with two repeats, followed by final training on all complete pairs and imputation of the missing blocks. Scaling and PCA are fitted on training observations only. The same algorithm and settings are fitted separately within each dataset, without requiring shared feature names between datasets.

Both methods use 15 PCA components per modality. KNN uses five distance-weighted neighbors. The Gaussian model uses four Gibbs chains with 2,000 warmup iterations and at most 20,000 iterations per chain. Diagnostics start at 6,000 iterations and repeat every 2,000. Convergence requires two consecutive checks with maximum R-hat <= 1.01 and minimum bulk and tail ESS >= 400. Nonconvergence is recorded explicitly.

Original-feature predictions are averaged over all retained posterior predictive states before computing distances. Features exactly constant in training are restored to their exact value after inverse PCA and after posterior averaging. This prevents numerical roundoff from being amplified by subsequent scaling. No features are removed and observed values are preserved. Gaussian and KNN reconstructions of varying features can contain negative values.

Validation masks one modality in 10%, 20%, 30%, 40% or 50% of held-out samples. Both methods use identical masks. Feature RMSE and predictive interval coverage evaluate hidden values only. Distribution figures use a balanced subset of 200 predictive draws.

`official_style_r` compares the true and completed held-out distance matrices with scaling and PCA fitted separately to each table. `mantel_r` uses one fixed distance space fitted on training data. Both include all sample pairs and are correlations, not percentages of correct imputations. `affected_pairs_r` includes only pairs involving an imputed sample. Fold standard deviations are descriptive, not independent-sample confidence intervals.

## Outputs

Results are saved in `outputs/name/`. The Gaussian submission matrix is `outputs/name/final/shira_res_name.csv`. It includes all samples, with identifiers in rows and columns, comma delimiters and full numeric precision. Final distances are Euclidean distances in the first two PCs of the combined standardized completed omics data. Change the submission prefix with `--student`.

Outputs also include the KNN distance matrix, validation summaries, feature errors, distribution figures, convergence diagnostics and run configuration. Raw posterior draws, completed feature tables and intermediate distance matrices are not saved. Compress the complete `outputs` directory to share results.

## Options

Append `--skip-validation` for final predictions only, or `--skip-final` for validation only. Append `--resume` to skip completed folds from a run with identical code, inputs and configuration. An unfinished fold restarts. Do not resume outputs from an earlier code version. Use `--output` to select a new output directory. Run with `--help` for all options.
