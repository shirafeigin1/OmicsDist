#!/usr/bin/env python3
"""Command-line interface for multi-omics distance estimation."""

import os

# Configure numerical library threads before importing NumPy.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
from pathlib import Path
import sys

from multiomics_distance_pipeline.config import Config
from multiomics_distance_pipeline.data_loading import (
    load_tables,
    data_structure_summary,
)
from multiomics_distance_pipeline.validation import run_validation_experiment
from multiomics_distance_pipeline.final_run import run_final_matrices


def parse_args():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description="OmicsDist: PCA-KNN and Gaussian Gibbs regression."
    )
    p.add_argument(
        "--dataset", required=True, metavar="name", help="Dataset folder name under data/."
    )
    p.add_argument("--student", default="shira", help="Submission filename prefix.")
    for name in ["metadata", "microbiome", "metabolome"]:
        p.add_argument("--" + name, type=Path)
    p.add_argument("--output", type=Path)
    for name in [
        "n_components",
        "knn_neighbors",
        "n_chains",
        "burn_in",
        "min_iter",
        "max_iter",
        "check_every",
        "thin",
        "ess_min",
        "predictive_draws",
        "folds",
        "repeats",
        "permutations",
        "seed",
    ]:
        p.add_argument(
            "--" + name.replace("_", "-"), type=int, default=getattr(Config(), name)
        )
    for name in ["rhat_limit", "tau2", "a0", "b0"]:
        p.add_argument(
            "--" + name.replace("_", "-"), type=float, default=getattr(Config(), name)
        )
    p.add_argument("--micro-scale", choices=["auto", "relative", "clr"], default="auto")
    p.add_argument(
        "--missing-rates", nargs="+", type=float, default=list(Config().missing_rates)
    )
    p.add_argument("--skip-validation", action="store_true")
    p.add_argument("--skip-final", action="store_true")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed folds/final run with identical config, code and input hashes.",
    )
    args = p.parse_args()
    if not args.student or not all(c.isascii() and (c.isalnum() or c in "_-") for c in args.student):
        p.error("Student prefix must contain only ASCII letters, digits, underscores or hyphens.")
    if (
        not args.dataset.strip()
        or args.dataset in (".", "..")
        or not all(c.isalnum() or c in "_-. " for c in args.dataset)
        or args.dataset.endswith((".", " "))
    ):
        p.error("Dataset must be a single folder name using letters, digits, spaces, underscores, hyphens or dots.")
    data_dir = root / "data" / args.dataset
    for name in ["metadata", "microbiome", "metabolome"]:
        if getattr(args, name) is None:
            setattr(args, name, data_dir / f"{name}.csv")
    for name in ["metadata", "microbiome", "metabolome"]:
        if not getattr(args, name).is_file():
            p.error(f"Input file not found: {getattr(args, name)}")
    if args.output is None:
        args.output = root / "outputs" / args.dataset
    return args


def main():
    args = parse_args()
    cfg = Config(
        **{name: getattr(args, name) for name in Config.__dataclass_fields__}
    ).validate()
    if args.skip_validation and args.skip_final:
        raise ValueError("Both phases cannot be skipped.")
    args.output.mkdir(parents=True, exist_ok=True)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    hashes = {
        name: sha(getattr(args, name))
        for name in ["metadata", "microbiome", "metabolome"]
    }
    code_hash = hashlib.sha256(
        "".join(sha(p) for p in sorted(Path(__file__).parent.rglob("*.py"))).encode()
    ).hexdigest()
    signature = {"config": cfg.dict(), "input_sha256": hashes, "code_sha256": code_hash, "dataset": args.dataset, "student": args.student}
    signature = json.loads(json.dumps(signature))
    manifest = args.output / "run_manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if not args.resume:
            raise ValueError(
                "Output already contains a run. Use --resume or a new --output folder."
            )
        if previous["signature"] != signature:
            raise ValueError(
                "Resume refused: configuration, inputs or source code changed. Use a new output directory."
            )
    elif args.resume:
        raise ValueError("No previous run manifest exists at this output path.")
    else:
        versions = {
            n: importlib.metadata.version(n)
            for n in ["numpy", "pandas", "scipy", "scikit-learn", "arviz", "matplotlib"]
        }
        manifest.write_text(
            json.dumps(
                {
                    "signature": signature,
                    "python": sys.version,
                    "platform": platform.platform(),
                    "versions": versions,
                },
                indent=2,
            )
        )
    logging.captureWarnings(True)
    logging.basicConfig(
        filename=args.output / "warnings.log",
        level=logging.WARNING,
        format="%(asctime)s %(message)s",
    )
    data = load_tables(args.metadata, args.microbiome, args.metabolome)
    data_structure_summary(data).to_csv(
        args.output / "data_structure_summary.csv", index=False
    )
    print("OmicsDist: cross-validation and distance estimation.", flush=True)
    if not args.skip_validation:
        run_validation_experiment(data, args.output, cfg, args.resume)
    if not args.skip_final:
        run_final_matrices(data, args.output, cfg, args.resume, args.dataset, args.student)
    print(f"Completed. Output directory: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
