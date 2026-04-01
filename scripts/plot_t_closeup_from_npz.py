#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys
import argparse

# ============================================================
# Unified CVD-safe color palette for all JetPrism figures  
# ============================================================
COLOR_TRUTH = '#0072B2'       # Blue - reference/truth data
COLOR_GENERATED = '#E69F00'   # Orange/amber - model output
COLOR_RATIO = '#404040'       # Dark gray - ratio lines
COLOR_CONTEXT = '#808080'     # Medium gray - reference lines/grid
COLOR_BAND = '#CCCCCC'        # Light gray - uncertainty bands
COLOR_MARKER = '#D55E00'      # Vermillion - boundary markers

# ============================================================
# JINST-compatible Matplotlib settings
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.dpi": 300,
})

def main():
    parser = argparse.ArgumentParser(description="Plot t channel close-ups from npz")
    parser.add_argument("run_dir", help="Path to run directory")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    print(f"Run directory: {run_dir}")
    
    # Load generated data
    npz_path = run_dir / "generated_samples_best.npz"
    if not npz_path.exists():
        npz_path = run_dir / "generated_samples.npz"
    print(f"Loading generated data from {npz_path}")
    gen_data = np.load(npz_path)
    gen_t = gen_data["samples"][:, 0]
    
    # Load truth data
    cache_path = run_dir / "dataset_cache.npz"
    print(f"Loading truth data from {cache_path}")
    cache_data = np.load(cache_path, allow_pickle=True)
    truth_t = cache_data["pre_transform_data"][:, 0]
    
    print(f"Truth t N={len(truth_t):,}")
    print(f"Gen t N={len(gen_t):,}")
    
    # Create figure with ratio subplots using GridSpec
    fig = plt.figure(figsize=(12, 7))
    
    # 2 columns, each with main plot (3 parts) and ratio (1 part)
    gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=0.0, wspace=0.25)
    
    ultra_bins = 200
    
    # 1. Close-up near -0.4 (upper bound)
    ax_main1 = fig.add_subplot(gs[0, 0])
    ax_ratio1 = fig.add_subplot(gs[1, 0], sharex=ax_main1)
    
    closeup_range_upper = (-0.42, -0.38)
    
    counts_t1, bins1 = np.histogram(truth_t, bins=ultra_bins, range=closeup_range_upper, density=True)
    counts_g1, _ = np.histogram(gen_t, bins=ultra_bins, range=closeup_range_upper, density=True)
    
    ax_main1.fill_between(bins1[:-1], counts_t1, step="post", alpha=0.6, color=COLOR_TRUTH, label='Ground truth')
    ax_main1.step(bins1[:-1], counts_g1, where="post", color=COLOR_GENERATED, linewidth=1.5, label='Generated')
    ax_main1.set_ylabel('Density')
    ax_main1.legend()
    ax_main1.axvline(-0.4, color=COLOR_MARKER, linestyle='--', alpha=0.8)
    ax_main1.set_xlim(closeup_range_upper)
    ax_main1.grid(True, linestyle=':', alpha=0.5)
    plt.setp(ax_main1.get_xticklabels(), visible=False)
    
    # Ratio subplot 1
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio1 = np.where(counts_t1 > 0, counts_g1 / counts_t1, np.nan)
    ax_ratio1.step(bins1[:-1], ratio1, where="post", color=COLOR_RATIO, linewidth=1.0, linestyle='-')
    ax_ratio1.axhline(1.0, color=COLOR_CONTEXT, linestyle='--', linewidth=0.8)
    ax_ratio1.fill_between(bins1[:-1], 0.9, 1.1, alpha=0.2, color=COLOR_BAND, step="post")
    ax_ratio1.set_xlabel(r'$t$')
    ax_ratio1.set_ylabel('Gen/Truth')
    ax_ratio1.set_ylim(0.5, 1.5)
    ax_ratio1.grid(True, linestyle=':', alpha=0.5)
    
    # 2. Close-up near -1.0 (lower bound)
    ax_main2 = fig.add_subplot(gs[0, 1])
    ax_ratio2 = fig.add_subplot(gs[1, 1], sharex=ax_main2)
    
    closeup_range_lower = (-1.02, -0.98)
    
    counts_t2, bins2 = np.histogram(truth_t, bins=ultra_bins, range=closeup_range_lower, density=True)
    counts_g2, _ = np.histogram(gen_t, bins=ultra_bins, range=closeup_range_lower, density=True)
    
    ax_main2.fill_between(bins2[:-1], counts_t2, step="post", alpha=0.6, color=COLOR_TRUTH, label='Ground truth')
    ax_main2.step(bins2[:-1], counts_g2, where="post", color=COLOR_GENERATED, linewidth=1.5, label='Generated')
    ax_main2.legend()
    ax_main2.axvline(-1.0, color=COLOR_MARKER, linestyle='--', alpha=0.8)
    ax_main2.set_xlim(closeup_range_lower)
    ax_main2.grid(True, linestyle=':', alpha=0.5)
    plt.setp(ax_main2.get_xticklabels(), visible=False)
    
    # Ratio subplot 2
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio2 = np.where(counts_t2 > 0, counts_g2 / counts_t2, np.nan)
    ax_ratio2.step(bins2[:-1], ratio2, where="post", color=COLOR_RATIO, linewidth=1.0, linestyle='-')
    ax_ratio2.axhline(1.0, color=COLOR_CONTEXT, linestyle='--', linewidth=0.8)
    ax_ratio2.fill_between(bins2[:-1], 0.9, 1.1, alpha=0.2, color=COLOR_BAND, step="post")
    ax_ratio2.set_xlabel(r'$t$')
    ax_ratio2.set_ylabel('Gen/Truth')
    ax_ratio2.set_ylim(0.5, 1.5)
    ax_ratio2.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    # Save figures
    output_path_pdf = run_dir / "t_channel_closeup_highres.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Saved: {output_path_pdf}")

if __name__ == '__main__':
    main()
