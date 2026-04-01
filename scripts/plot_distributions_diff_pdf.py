#!/usr/bin/env python
"""Generate PDF version of distributions_diff plot from saved npz files."""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jetprism.utils import _MCPOM_COLS_LIST, _MCPOM_SCALES, _MCPOM_LABELS, _MCPOM_COL_INDEX

# ============================================================
# Unified CVD-safe color palette for all JetPrism figures
# ============================================================
COLOR_TRUTH = '#0072B2'       # Blue - reference/truth data
COLOR_GENERATED = '#E69F00'   # Orange/amber - model output
COLOR_DETECTOR = '#CC79A7'    # Pink/magenta - detector/degraded
COLOR_RATIO = '#404040'       # Dark gray - ratio lines
COLOR_CONTEXT = '#808080'     # Medium gray - reference lines/grid
COLOR_BAND = '#CCCCCC'        # Light gray - uncertainty bands

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


def plot_distributions_diff_1d_pdf(truth, generated, output_path, label_a='Ground truth', label_b='Generated'):
    """Plot overlaid 1-D distributions for truth vs generated (mock datasets) and save as PDF.

    Args:
        truth:       Numpy array (will be flattened to 1-D).
        generated:   Numpy array (will be flattened to 1-D).
        output_path: Full path to save the plot.
        label_a:     Legend label for truth data.
        label_b:     Legend label for generated data.
    """
    from matplotlib.gridspec import GridSpec
    
    truth_flat = np.array(truth).flatten()
    gen_flat = np.array(generated).flatten()

    # Compute shared bin edges based on full data range (no percentile clipping)
    combined = np.concatenate([truth_flat, gen_flat])
    vmin, vmax = np.min(combined), np.max(combined)
    bins = np.linspace(vmin, vmax, 201)

    # Create figure with GridSpec for main plot and ratio subplot
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    
    ax_main = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

    # Plot truth
    counts_t, _ = np.histogram(truth_flat, bins=bins, density=True)
    ax_main.fill_between(bins[:-1], counts_t, step="post", alpha=0.5, color=COLOR_TRUTH, label=label_a)

    # Plot generated
    counts_g, _ = np.histogram(gen_flat, bins=bins, density=True)
    ax_main.step(bins[:-1], counts_g, where="post", color=COLOR_GENERATED, linewidth=1.5, label=label_b)

    ax_main.set_ylabel('Density')
    ax_main.legend()
    ax_main.grid(True, linestyle=':')
    plt.setp(ax_main.get_xticklabels(), visible=False)
    
    # Compute and plot ratio (Generated / Truth)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(counts_t > 0, counts_g / counts_t, np.nan)
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax_ratio.step(bins[:-1], ratio, where="post", color=COLOR_RATIO, linewidth=1.2, linestyle='-')
    ax_ratio.axhline(1.0, color=COLOR_CONTEXT, linestyle='--', linewidth=1)
    ax_ratio.fill_between(bins[:-1], 0.9, 1.1, alpha=0.2, color=COLOR_BAND)
    ax_ratio.set_xlabel('Value')
    ax_ratio.set_ylabel('Gen / Truth')
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.grid(True, linestyle=':')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, format='pdf', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved 1D PDF plot to {output_path}")


def plot_distributions_diff_pdf(dataset_a, dataset_b, output_filepath, label_a='Ground truth', label_b='Generated'):
    """Plot an overlay comparison of two MC-POM datasets and save as PDF.
    
    Args:
        dataset_a: First dataset (truth), [N, 24] array.
        dataset_b: Second dataset (generated), [M, 24] array.
        output_filepath: Path for the saved figure (should end with .pdf).
        label_a: Legend label for first dataset.
        label_b: Legend label for second dataset.
    """
    from matplotlib.gridspec import GridSpec
    
    cols_to_plot = _MCPOM_COLS_LIST
    scales = _MCPOM_SCALES
    num_bins = 200
    
    nrows = 5
    ncols = 4
    
    # Create figure with GridSpec: each column pair has main plot (3 parts) + ratio (1 part)
    fig = plt.figure(figsize=(20, 18))
    
    # Create outer grid for the 5x4 layout with reduced bottom margin for legend
    outer_gs = GridSpec(nrows, ncols, figure=fig, hspace=0.33, wspace=0.2,
                        bottom=0.08, top=0.95, left=0.05, right=0.98)
    
    main_axes = []
    ratio_axes = []

    for i, col in enumerate(cols_to_plot):
        row = i // ncols
        col_idx = i % ncols
        
        # Create inner gridspec for each subplot (main + ratio)
        inner_gs = outer_gs[row, col_idx].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax_main = fig.add_subplot(inner_gs[0])
        ax_ratio = fig.add_subplot(inner_gs[1], sharex=ax_main)
        
        main_axes.append(ax_main)
        ratio_axes.append(ax_ratio)
        
        # Use index mapping to get correct column from full 24-column data
        data_idx = _MCPOM_COL_INDEX[col]
        data_a = dataset_a[:, data_idx]
        data_b = dataset_b[:, data_idx]

        if col in scales and 'xlim' in scales[col]:
            bin_range = scales[col]['xlim']
        else:
            combined_min = min(np.min(data_a), np.min(data_b))
            combined_max = max(np.max(data_a), np.max(data_b))
            bin_range = (combined_min, combined_max)

        bins = np.linspace(bin_range[0], bin_range[1], num_bins + 1)

        counts_a, bin_edges = np.histogram(data_a, bins=bins, density=True)
        counts_b, _ = np.histogram(data_b, bins=bins, density=True)

        # Main distribution plot
        ax_main.fill_between(bin_edges[:-1], counts_a, step="post",
                        alpha=0.5, color=COLOR_TRUTH, label=label_a)
        ax_main.step(bin_edges[:-1], counts_b, where="post",
                color=COLOR_GENERATED, linewidth=1.5, linestyle='-', label=label_b)
        ax_main.fill_between(bin_edges[:-1], counts_b, step="post",
                        alpha=0.15, color=COLOR_GENERATED)

        label = _MCPOM_LABELS.get(col, col)
        ax_main.set_title(label)
        ax_main.set_ylabel('Density')
        ax_main.grid(True, linestyle=':', linewidth=0.5)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        if col in scales and 'xlim' in scales[col]:
            ax_main.set_xlim(scales[col]['xlim'])
        
        # Ratio subplot (Generated / Truth)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(counts_a > 0, counts_b / counts_a, np.nan)
        
        ax_ratio.step(bin_edges[:-1], ratio, where="post", color=COLOR_RATIO, linewidth=1.0, linestyle='-')
        ax_ratio.axhline(1.0, color=COLOR_CONTEXT, linestyle='--', linewidth=0.8)
        ax_ratio.fill_between(bin_edges[:-1], 0.9, 1.1, alpha=0.2, color=COLOR_BAND, step="post")
        ax_ratio.set_xlabel('Value')
        ax_ratio.set_ylabel('Ratio')
        ax_ratio.set_ylim(0.5, 1.5)
        ax_ratio.grid(True, linestyle=':', linewidth=0.5)

    handles, labels = main_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, 0.02))
    
    os.makedirs(os.path.dirname(output_filepath) or '.', exist_ok=True)
    plt.savefig(output_filepath, format='pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved PDF plot to {output_filepath}")


def main():
    parser = argparse.ArgumentParser(description='Generate PDF version of distributions_diff plot')
    parser.add_argument('run_dir', type=str, help='Path to run directory (e.g., outputs/2026-03-20/00-31-10_eujlglkx)')
    parser.add_argument('--suffix', type=str, default='best', help='Suffix for the samples file (default: best)')
    parser.add_argument('--output', type=str, default=None, help='Output PDF path (default: <run_dir>/distributions_diff_<suffix>.pdf)')
    args = parser.parse_args()

    run_dir = args.run_dir.rstrip('/')
    suffix = args.suffix
    
    # Load generated samples
    gen_path = os.path.join(run_dir, f'generated_samples_{suffix}.npz')
    if not os.path.exists(gen_path):
        print(f"Error: Generated samples not found at {gen_path}")
        sys.exit(1)
    
    print(f"Loading generated samples from {gen_path}...")
    gen_data = np.load(gen_path)
    generated = gen_data['samples']
    print(f"  Generated shape: {generated.shape}")
    
    # Load truth data from dataset cache
    cache_path = os.path.join(run_dir, 'dataset_cache.npz')
    if not os.path.exists(cache_path):
        print(f"Error: Dataset cache not found at {cache_path}")
        sys.exit(1)
    
    print(f"Loading truth data from {cache_path}...")
    cache_data = np.load(cache_path, allow_pickle=True)
    truth = cache_data['pre_transform_data']
    print(f"  Truth shape: {truth.shape}")
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(run_dir, f'distributions_diff_{suffix}.pdf')
    
    # Generate PDF plot - detect 1D (mock) vs multi-D (MC-POM)
    print(f"Generating PDF plot...")
    is_1d = (len(generated.shape) == 1) or (generated.shape[1] == 1)
    if is_1d:
        plot_distributions_diff_1d_pdf(truth, generated, output_path)
    else:
        plot_distributions_diff_pdf(truth, generated, output_path)


if __name__ == '__main__':
    main()
