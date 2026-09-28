"""Rank-normalized split R-hat, effective sample size and Monte Carlo error."""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize(draws):
    idata = az.from_dict(posterior=draws)
    # No rounding before threshold checks.
    return az.summary(idata, kind="all", round_to="none")


def run_chains(factory, cfg, seed, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    chains = [factory(seed + 1009 * c) for c in range(cfg.n_chains)]
    stored = [{k: [] for k in c.values()} for c in chains]
    history = []
    passed_twice = 0
    end = 0
    passed = False
    started = time.monotonic()
    last_progress = started
    for start in range(0, cfg.max_iter, cfg.check_every):
        end = min(start + cfg.check_every, cfg.max_iter)
        for c, chain in enumerate(chains):
            for it in range(start, end):
                chain.step(it)
                if (it + 1) % 100 == 0 and time.monotonic() - last_progress >= 30:
                    elapsed = time.monotonic() - started
                    print(
                        f"  {out.name}: chain {c + 1}/{cfg.n_chains}, "
                        f"iteration {it + 1}/{cfg.max_iter}, elapsed {elapsed:.0f}s",
                        flush=True,
                    )
                    last_progress = time.monotonic()
                if it >= cfg.burn_in and (it - cfg.burn_in) % cfg.thin == 0:
                    for key, val in chain.values().items():
                        stored[c][key].append(val)
        if end < cfg.min_iter:
            print(f"  {out.name}: warmup/sampling {end}/{cfg.max_iter}", flush=True)
            continue
        draws = {k: np.asarray([s[k] for s in stored]) for k in stored[0]}
        summary = summarize(draws)
        numeric = summary[["r_hat", "ess_bulk", "ess_tail"]].to_numpy()
        passed = bool(
            np.all(np.isfinite(numeric))
            and summary.r_hat.max() <= cfg.rhat_limit
            and summary.ess_bulk.min() >= cfg.ess_min
            and summary.ess_tail.min() >= cfg.ess_min
        )
        passed_twice = passed_twice + 1 if passed else 0
        history.append(
            {
                "iterations": end,
                "max_rhat": summary.r_hat.max(),
                "min_ess_bulk": summary.ess_bulk.min(),
                "min_ess_tail": summary.ess_tail.min(),
                "passes_thresholds": passed,
            }
        )
        pd.DataFrame(history).to_csv(out / "convergence_history.csv", index=False)
        summary.to_csv(out / "parameter_summary.csv", index_label="parameter")
        print(
            f"  {out.name}: {end} iterations | Rhat max={summary.r_hat.max():.4f} | "
            f"ESS min={summary.ess_bulk.min():.0f}/{summary.ess_tail.min():.0f}",
            flush=True,
        )
        if passed_twice >= 2:
            break
    # Convergence requires two consecutive successful diagnostic checks.
    status = {
        "converged": passed_twice >= 2,
        "last_check_passed": passed,
        "iterations": end,
        "saved_per_chain": len(next(iter(stored[0].values()))),
        "stop_reason": (
            "two_consecutive_passes" if passed_twice >= 2 else "iteration_limit"
        ),
        "criteria": {
            "rhat_max": cfg.rhat_limit,
            "ess_bulk_min": cfg.ess_min,
            "ess_tail_min": cfg.ess_min,
        },
        "chains": [c.status() for c in chains],
    }
    (out / "convergence.json").write_text(json.dumps(status, indent=2))
    trace_plots(draws, summary, out)
    return draws, status


def trace_plots(draws, summary, out):
    flat = np.concatenate(
        [v.reshape(v.shape[0], v.shape[1], -1) for v in draws.values()], axis=-1
    )
    # az.summary follows insertion and C-order flattening of the posterior variables.
    names = summary.index.to_list()
    worst = np.argsort(np.nan_to_num(summary.r_hat.to_numpy(), nan=np.inf))[-3:]
    selected = list(dict.fromkeys([0, flat.shape[-1] - 1, *worst]))
    fig, axes = plt.subplots(
        len(selected), 2, figsize=(12, 2.2 * len(selected)), squeeze=False
    )
    for row, j in enumerate(selected):
        for c in range(flat.shape[0]):
            axes[row, 0].plot(flat[c, :, j], alpha=0.7, lw=0.7, label=f"chain {c+1}")
            axes[row, 1].hist(
                flat[c, :, j], bins=35, density=True, histtype="step", alpha=0.8
            )
        axes[row, 0].set_title(names[j])
        axes[row, 0].set_xlabel("Retained draw")
        axes[row, 1].set_title("Posterior density")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "trace_and_posterior.png", dpi=160)
    plt.close(fig)
