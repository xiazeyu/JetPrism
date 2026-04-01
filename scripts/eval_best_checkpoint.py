#!/usr/bin/env python3
"""
Evaluate metrics on the BEST checkpoint using pre-generated samples.

Uses generated_samples_best.npz (already produced by PREDICT mode) so no
GPU or model loading is needed — runs on a CPU node.

Computes: chi2, W1, chi2_2D, D_corr, R_NN

Usage:
    source .venv/bin/activate
    python scripts/eval_best_checkpoint.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Runs to evaluate ─────────────────────────────────────────────────────────

# # MC-POM generation & denoising runs
# RUNS = {
#     "mc_pom_gen": {
#         "path": "outputs/2026-03-20/00-31-10_eujlglkx",
#         "label": "Generation",
#         "sigma": "---",
#     },
#     "mc_pom_denoise_sigma2.0": {
#         "path": "outputs/2026-03-22/04-15-03_2ah1x9hh",
#         "label": "Denoise",
#         "sigma": "2.0",
#     },
#     "mc_pom_denoise_sigma1.0": {
#         "path": "outputs/2026-03-22/04-15-03_60ido4bi",
#         "label": "Denoise",
#         "sigma": "1.0",
#     },
#     "mc_pom_denoise_sigma0.5": {
#         "path": "outputs/2026-03-22/04-15-05_drr3p7zs",
#         "label": "Denoise",
#         "sigma": "0.5",
#     },
# }

# Mock generation runs (1D synthetic benchmarks)
RUNS = {
    "bimodal_asym": {
        "path": "outputs/2026-03-22/21-40-05_1dlviacs",
        "label": "bimodal_asym",
        "sigma": "---",
    },
    "delta_0": {
        "path": "outputs/2026-03-22/21-40-05_bkrx6brp",
        "label": "delta_0",
        "sigma": "---",
    },
    "exponential_decay": {
        "path": "outputs/2026-03-22/21-40-05_sw2p407u",
        "label": "exp_decay",
        "sigma": "---",
    },
    "gauss_cutoff": {
        "path": "outputs/2026-03-22/21-40-05_u9qmfdkx",
        "label": "gauss_cutoff",
        "sigma": "---",
    },
    "narrow_wide_overlap": {
        "path": "outputs/2026-03-22/21-40-05_6aijfyng",
        "label": "narrow_wide",
        "sigma": "---",
    },
    "noise_3spikes": {
        "path": "outputs/2026-03-22/21-40-05_ggo2lzyf",
        "label": "noise_3spk",
        "sigma": "---",
    },
    "noise_10spikes": {
        "path": "outputs/2026-03-22/21-40-05_c8opna4i",
        "label": "noise_10spk",
        "sigma": "---",
    },
    "tall_flat_far": {
        "path": "outputs/2026-03-22/21-40-05_15aedvbl",
        "label": "tall_flat",
        "sigma": "---",
    },
    "triple_flat_spread": {
        "path": "outputs/2026-03-22/21-40-05_rrzmwkhu",
        "label": "triple_flat",
        "sigma": "---",
    },
    "triple_mixed": {
        "path": "outputs/2026-03-22/21-40-05_54jfcj9a",
        "label": "triple_mixed",
        "sigma": "---",
    },
    "uniform_flat": {
        "path": "outputs/2026-03-22/21-40-05_50am7k4i",
        "label": "uniform_flat",
        "sigma": "---",
    },
}

# Evaluation settings (match training validation)
METRIC_SAMPLE_SIZE = 1_000_000
NN_SAMPLE_SIZE = 80_000
NN_QUERY_BATCH = 1_000
NN_REF_BATCH = 50_000


def load_data(run_dir: Path):
    """Load pre-generated samples, transform, truth, and training data."""
    # Pre-generated samples (physical space, from PREDICT on best.ckpt)
    samples_path = run_dir / "generated_samples_best.npz"
    if not samples_path.exists():
        raise FileNotFoundError(f"No generated_samples_best.npz in {run_dir}")
    samples_physical = np.load(str(samples_path))["samples"]

    # Best checkpoint epoch
    ckpt_path = run_dir / "checkpoints" / "best.ckpt"
    best_epoch = "?"
    if ckpt_path.exists():
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        best_epoch = checkpoint.get("epoch", "?")

        # Transform (needed to forward-transform samples for NN metric)
        transform = None
        if "transform_state" in checkpoint:
            from jetprism.transforms import BaseTransform
            transform = BaseTransform.deserialize(checkpoint["transform_state"])
        del checkpoint
    else:
        transform = None

    # Dataset cache
    cache_path = run_dir / "dataset_cache.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"No dataset_cache.npz in {run_dir}")
    cache = np.load(str(cache_path), allow_pickle=True)

    # Truth data (physical space)
    if "pre_transform_data" in cache:
        truth_physical = cache["pre_transform_data"]
    elif "data" in cache and transform is not None:
        truth_physical = transform.inverse_transform(cache["data"])
    else:
        truth_physical = cache["data"]

    # Training data in transformed space (for NN metric)
    training_transformed = cache["data"] if "data" in cache else None

    # Config
    config_path = run_dir / ".hydra" / "config.yaml"
    cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    is_paired = cfg.get("dataset", {}).get("paired", False)

    return samples_physical, truth_physical, training_transformed, transform, is_paired, best_epoch


def compute_metrics(run_name, run_info, device):
    """Compute all 5 metrics for a single run using pre-generated samples."""
    from scipy.stats import ks_2samp, wasserstein_distance
    from jetprism.metric import (
        chi2_metric,
        compute_joint_distribution_metrics,
        compute_nn_memorization_metric,
    )

    run_dir = PROJECT_ROOT / run_info["path"]
    print(f"\n{'='*60}")
    print(f"Evaluating: {run_name} ({run_dir.name})")
    print(f"{'='*60}")

    samples_physical, truth_physical, training_transformed, transform, is_paired, best_epoch = load_data(run_dir)
    print(f"  Best checkpoint epoch: {best_epoch}")
    print(f"  Loaded samples: {samples_physical.shape}, truth: {truth_physical.shape}")

    n = min(METRIC_SAMPLE_SIZE, len(truth_physical), len(samples_physical))
    truth = truth_physical[:n]
    samples = samples_physical[:n]
    print(f"  Using {n} samples for metrics")

    if truth.ndim == 1:
        truth = truth[:, None]
    if samples.ndim == 1:
        samples = samples[:, None]

    # ── Marginal metrics (chi2, KS, W1) ──────────────────────────────────────
    num_features = truth.shape[1]
    chi2_vals, ks_vals, w1_vals = [], [], []
    for i in range(num_features):
        t, g = truth[:, i], samples[:, i]
        chi2_vals.append(float(chi2_metric(t, g)))
        ks_vals.append(float(ks_2samp(t, g).statistic))
        w1_vals.append(float(wasserstein_distance(t, g)))

    chi2_mean = float(np.mean(chi2_vals))
    ks_mean = float(np.mean(ks_vals))
    w1_mean = float(np.mean(w1_vals))
    print(f"  chi2_mean={chi2_mean:.2f}, ks_mean={ks_mean:.6f}, w1_mean={w1_mean:.6f}")

    # ── Joint metrics (chi2_2D, D_corr) ──────────────────────────────────────
    chi2_2d_mean = None
    corr_dist = None
    if num_features > 1:
        joint = compute_joint_distribution_metrics(truth, samples)
        chi2_2d_mean = joint["chi2_2d_mean"]
        corr_dist = joint["correlation_distance"]
        print(f"  chi2_2d_mean={chi2_2d_mean:.2f}, corr_dist={corr_dist:.6f}")

    # ── NN memorization ratio ────────────────────────────────────────────────
    # NN metric operates in transformed (model-output) space.
    # We forward-transform the physical-space samples back to transformed space.
    nn_ratio = None
    d_gen_to_train = None
    d_train_to_train = None
    if not is_paired and training_transformed is not None:
        if transform is not None and hasattr(transform, "transform"):
            samples_transformed = transform.transform(samples)
        else:
            samples_transformed = samples
        print(f"  Computing NN memorization metric ({NN_SAMPLE_SIZE} samples)...")
        nn_results = compute_nn_memorization_metric(
            generated=samples_transformed,
            training=training_transformed,
            nn_sample_size=NN_SAMPLE_SIZE,
            query_batch_size=NN_QUERY_BATCH,
            ref_batch_size=NN_REF_BATCH,
            device=device,
            rng_seed=42,
        )
        d_gen_to_train = nn_results["D_gen_to_train_mean"]
        d_train_to_train = nn_results["D_train_to_train_mean"]
        if d_train_to_train > 0:
            nn_ratio = d_gen_to_train / d_train_to_train
            print(f"  D_gen→train={d_gen_to_train:.6f}, D_train→train={d_train_to_train:.6f}, R_NN={nn_ratio:.4f}")
        else:
            nn_ratio = float("nan")
            print(f"  D_gen→train={d_gen_to_train:.6f}, D_train→train=0, R_NN=nan")

    return {
        "run_name": run_name,
        "label": run_info["label"],
        "sigma": run_info["sigma"],
        "best_epoch": best_epoch,
        "chi2_mean": chi2_mean,
        "ks_mean": ks_mean,
        "w1_mean": w1_mean,
        "chi2_2d_mean": chi2_2d_mean,
        "corr_dist": corr_dist,
        "d_gen_to_train": d_gen_to_train,
        "d_train_to_train": d_train_to_train,
        "nn_ratio": nn_ratio,
        "n_samples": n,
    }


def format_sci(val):
    if val is None:
        return "---"
    exp = int(f"{val:.2e}".split("e")[1])
    mantissa = val / (10**exp)
    return f"{mantissa:.2f}e{exp}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: NN memorization metric will be slow on CPU. "
              "Consider submitting to a GPU node for the NN computation.")

    results = []
    for run_name, run_info in RUNS.items():
        r = compute_metrics(run_name, run_info, device)
        results.append(r)

    # ── Print summary table ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("BEST CHECKPOINT EVALUATION RESULTS")
    print(f"{'='*80}")
    header = (f"{'Task':<12} {'σ':>4} {'Epoch':>6} {'χ²':>8} {'W1':>12} "
             f"{'χ²_2D':>8} {'D_corr':>12} {'D_g→t':>12} {'D_t→t':>12} {'R_NN':>8}")
    print(header)
    print("-" * len(header))
    for r in results:
        nn_str = f"{r['nn_ratio']:.2f}" if r["nn_ratio"] is not None else "---"
        chi2_2d_str = f"{r['chi2_2d_mean']:.1f}" if r["chi2_2d_mean"] is not None else "---"
        corr_str = format_sci(r["corr_dist"]) if r["corr_dist"] is not None else "---"
        dgt_str = format_sci(r["d_gen_to_train"]) if r["d_gen_to_train"] is not None else "---"
        dtt_str = format_sci(r["d_train_to_train"]) if r["d_train_to_train"] is not None else "---"
        print(
            f"{r['label']:<12} {r['sigma']:>4} {r['best_epoch']:>6} "
            f"{r['chi2_mean']:>8.1f} {format_sci(r['w1_mean']):>12} "
            f"{chi2_2d_str:>8} {corr_str:>12} {dgt_str:>12} {dtt_str:>12} {nn_str:>8}"
        )

    # ── Save JSON ────────────────────────────────────────────────────────────
    out_path = PROJECT_ROOT / "figures" / "best_checkpoint_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
