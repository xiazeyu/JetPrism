#!/usr/bin/env python
"""Generate PDF version of distributions_comparison plot for denoising tasks."""

import argparse
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
from pathlib import Path

import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jetprism import utils
from jetprism.transforms import BaseTransform
from jetprism.utils import _MCPOM_COLS_LIST, _MCPOM_SCALES, _MCPOM_LABELS, _MCPOM_COL_INDEX

# ============================================================
# Unified CVD-safe color palette for all JetPrism figures
# ============================================================
COLOR_TRUTH = '#0072B2'       # Blue - reference/truth data (Original)
COLOR_GENERATED = '#E69F00'   # Orange/amber - model output (Generated)
COLOR_DETECTOR = '#CC79A7'    # Pink/magenta - detector/degraded
COLOR_RATIO = '#404040'       # Dark gray - ratio lines
COLOR_CONTEXT = '#808080'     # Medium gray - reference lines/grid
COLOR_BAND = '#CCCCCC'        # Light gray - uncertainty bands

# Ordered color list for denoising comparisons: Original, Detector, Unfolded
DENOISE_COLORS = {
    'Original': COLOR_TRUTH,
    'Detector': COLOR_DETECTOR,
    'Unfolded': COLOR_GENERATED,
}

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


def plot_distributions_multiple_with_ratio(datasets, output_filepath, *,
                                           cols: dict[str, int] | None = None,
                                           reference_label: str = 'Original'):
    """Plot and compare distributions of multiple datasets with ratio subplots.
    
    Args:
        datasets:        ``{label: array}`` dict.  Each array is ``[N, D]``.
        output_filepath: Path to save the output image.
        cols:            Optional ``{col_name: data_column_index}`` mapping.
        reference_label: Label of the reference dataset for ratio computation.
    """
    if cols is None:
        # Map column names to their indices in the full 24-column data
        cols = {name: _MCPOM_COL_INDEX[name] for name in _MCPOM_COLS_LIST}

    n_plots = len(cols)
    ncols_grid = 4
    nrows_grid = math.ceil(n_plots / ncols_grid)

    fig = plt.figure(figsize=(20, 4 * nrows_grid))
    
    # Create outer grid for the layout with reduced bottom margin for legend
    outer_gs = GridSpec(nrows_grid, ncols_grid, figure=fig, hspace=0.33, wspace=0.2,
                        bottom=0.08, top=0.95, left=0.05, right=0.98)

    scales = _MCPOM_SCALES
    num_bins = 200
    
    main_axes = []
    ratio_axes = []
    
    # Get reference dataset for ratio
    ref_data = datasets.get(reference_label)
    if ref_data is None:
        ref_data = list(datasets.values())[0]
        reference_label = list(datasets.keys())[0]

    for i, (col, data_idx) in enumerate(cols.items()):
        row = i // ncols_grid
        col_idx = i % ncols_grid
        
        # Create inner gridspec for each subplot (main + ratio)
        inner_gs = outer_gs[row, col_idx].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax_main = fig.add_subplot(inner_gs[0])
        ax_ratio = fig.add_subplot(inner_gs[1], sharex=ax_main)
        
        main_axes.append(ax_main)
        ratio_axes.append(ax_ratio)

        # Determine bin range across all datasets for this column
        if col in scales and 'xlim' in scales[col]:
            bin_range = scales[col]['xlim']
        else:
            all_data_for_col = [ds[:, data_idx] for ds in datasets.values()]
            if not all_data_for_col or all(len(d) == 0 for d in all_data_for_col):
                bin_range = (0, 1)
            else:
                combined_min = min(np.min(d) for d in all_data_for_col if len(d) > 0)
                combined_max = max(np.max(d) for d in all_data_for_col if len(d) > 0)
                if combined_min == combined_max:
                    bin_range = (combined_min - 0.5, combined_max + 0.5)
                else:
                    bin_range = (combined_min, combined_max)

        bins = np.linspace(bin_range[0], bin_range[1], num_bins + 1)
        
        # Compute reference histogram for ratio
        ref_counts, bin_edges = np.histogram(ref_data[:, data_idx], bins=bins, density=True)

        for k, (label, data) in enumerate(datasets.items()):
            # Use explicit CVD-safe colors based on label
            color = DENOISE_COLORS.get(label, COLOR_CONTEXT)
            column_data = data[:, data_idx]

            if len(column_data) > 0:
                counts, _ = np.histogram(column_data, bins=bins, density=True)
                linestyle = '-' if label == reference_label else '--'
                ax_main.fill_between(bin_edges[:-1], counts,
                                step="post", alpha=0.15, color=color)
                ax_main.step(bin_edges[:-1], counts, where="post", color=color,
                        linewidth=1.5, linestyle=linestyle, label=label)
                
                # Plot ratio for non-reference datasets
                if label != reference_label:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ratio = np.where(ref_counts > 0, counts / ref_counts, np.nan)
                    ax_ratio.step(bin_edges[:-1], ratio, where="post", 
                                 color=color, linewidth=1.0, linestyle='-', label=f'{label}/{reference_label}')

        label = _MCPOM_LABELS.get(col, col)
        ax_main.set_title(label)
        ax_main.set_ylabel('Density')
        ax_main.grid(True, linestyle=':', linewidth=0.5)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        if col in scales and 'xlim' in scales[col]:
            ax_main.set_xlim(scales[col]['xlim'])
        
        # Ratio subplot styling
        ax_ratio.axhline(1.0, color=COLOR_CONTEXT, linestyle='--', linewidth=0.8)
        ax_ratio.fill_between(bin_edges[:-1], 0.9, 1.1, alpha=0.2, color=COLOR_BAND, step="post")
        ax_ratio.set_xlabel('Value')
        ax_ratio.set_ylabel('Ratio')
        ax_ratio.set_ylim(0.5, 1.5)
        ax_ratio.grid(True, linestyle=':', linewidth=0.5)

    # Hide unused axes
    for j in range(n_plots, nrows_grid * ncols_grid):
        row = j // ncols_grid
        col_idx = j % ncols_grid
        if row < nrows_grid:
            inner_gs = outer_gs[row, col_idx].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
            ax_empty_main = fig.add_subplot(inner_gs[0])
            ax_empty_ratio = fig.add_subplot(inner_gs[1])
            ax_empty_main.set_visible(False)
            ax_empty_ratio.set_visible(False)

    handles, labels = main_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(datasets), 4),
               bbox_to_anchor=(0.5, 0.02))

    output_dir = os.path.dirname(output_filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison plot with ratio saved as {output_filepath}")


def plot_distributions_1d_with_ratio(datasets, output_filepath, reference_label='Original'):
    """Plot and compare 1D distributions with ratio subplot.
    
    Args:
        datasets:        ``{label: array}`` dict.  Each array is 1D or [N, 1].
        output_filepath: Path to save the output image.
        reference_label: Label of the reference dataset for ratio computation.
    """
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    
    ax_main = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)
    
    # Flatten all datasets and compute common bin range
    flat_data = {k: np.array(v).flatten() for k, v in datasets.items()}
    combined = np.concatenate(list(flat_data.values()))
    vmin, vmax = np.percentile(combined, [0.5, 99.5])
    bins = np.linspace(vmin, vmax, 201)
    
    # Get reference
    ref_data = flat_data.get(reference_label)
    if ref_data is None:
        ref_data = list(flat_data.values())[0]
        reference_label = list(flat_data.keys())[0]
    
    ref_counts, bin_edges = np.histogram(ref_data, bins=bins, density=True)
    
    for k, (label, data) in enumerate(flat_data.items()):
        # Use explicit CVD-safe colors based on label
        color = DENOISE_COLORS.get(label, COLOR_CONTEXT)
        counts, _ = np.histogram(data, bins=bins, density=True)
        linestyle = '-' if label == reference_label else '--'
        
        ax_main.fill_between(bin_edges[:-1], counts, step="post", alpha=0.15, color=color)
        ax_main.step(bin_edges[:-1], counts, where="post", color=color,
                    linewidth=1.5, linestyle=linestyle, label=label)
        
        if label != reference_label:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(ref_counts > 0, counts / ref_counts, np.nan)
            ax_ratio.step(bin_edges[:-1], ratio, where="post", 
                         color=color, linewidth=1.0, linestyle='-')
    
    ax_main.set_ylabel('Density')
    ax_main.set_title('Distribution Comparison')
    ax_main.legend()
    ax_main.grid(True, linestyle=':')
    plt.setp(ax_main.get_xticklabels(), visible=False)
    
    ax_ratio.axhline(1.0, color=COLOR_CONTEXT, linestyle='--', linewidth=0.8)
    ax_ratio.fill_between(bin_edges[:-1], 0.9, 1.1, alpha=0.2, color=COLOR_BAND, step="post")
    ax_ratio.set_xlabel('Value')
    ax_ratio.set_ylabel(f'X/{reference_label}')
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.grid(True, linestyle=':')
    
    os.makedirs(os.path.dirname(output_filepath) or '.', exist_ok=True)
    plt.savefig(output_filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved 1D comparison plot with ratio to {output_filepath}")


def load_transform_from_checkpoint(run_dir: str):
    """Load transform from checkpoint file.
    
    Args:
        run_dir: Path to run directory.
        
    Returns:
        Transform object or None if not found.
    """
    run_path = Path(run_dir)
    
    # Try to find checkpoint file
    candidates = [
        run_path / 'final_model.ckpt',
        run_path / 'checkpoints' / 'best.ckpt',
        run_path / 'checkpoints' / 'last.ckpt',
    ]
    
    ckpt_path = None
    for c in candidates:
        if c.exists():
            ckpt_path = c
            break
    
    if ckpt_path is None:
        print("Warning: No checkpoint found for loading transform")
        return None
    
    print(f"Loading transform from {ckpt_path}...")
    checkpoint = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    
    if 'transform_state' in checkpoint:
        transform = BaseTransform.deserialize(checkpoint['transform_state'])
        if transform is not None:
            print("  Loaded transform successfully")
            return transform
    
    print("Warning: No transform found in checkpoint")
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Generate PDF version of distributions_comparison plot for denoising tasks'
    )
    parser.add_argument(
        'run_dirs', 
        type=str, 
        nargs='+',
        help='Path(s) to run directory (e.g., outputs/2026-03-22/04-15-03_60ido4bi)'
    )
    parser.add_argument(
        '--suffix', 
        type=str, 
        default='best', 
        help='Suffix for the samples file (default: best)'
    )
    args = parser.parse_args()

    for run_dir in args.run_dirs:
        run_dir = run_dir.rstrip('/')
        suffix = args.suffix
        
        print(f"\n{'='*60}")
        print(f"Processing: {run_dir}")
        print(f"{'='*60}")
        
        # Load generated samples
        gen_path = os.path.join(run_dir, f'generated_samples_{suffix}.npz')
        if not os.path.exists(gen_path):
            print(f"Error: Generated samples not found at {gen_path}")
            continue
        
        print(f"Loading generated samples from {gen_path}...")
        gen_data = np.load(gen_path)
        samples = gen_data['samples']
        print(f"  Generated shape: {samples.shape}")
        
        # Load data from dataset cache
        cache_path = os.path.join(run_dir, 'dataset_cache.npz')
        if not os.path.exists(cache_path):
            print(f"Error: Dataset cache not found at {cache_path}")
            continue
        
        print(f"Loading data from {cache_path}...")
        cache_data = np.load(cache_path, allow_pickle=True)
        
        # Get number of samples to match
        n = len(samples)
        
        # Load truth (pre_transform_data for physical space)
        if 'pre_transform_data' in cache_data:
            truth_physical = cache_data['pre_transform_data'][:n]
            print(f"  Truth (pre_transform) shape: {truth_physical.shape}")
        else:
            print("Warning: pre_transform_data not found, using original_data")
            truth_physical = cache_data['original_data'][:n]
            print(f"  Truth (original) shape: {truth_physical.shape}")
        
        # Load detector data and inverse transform to physical space
        if 'detector_data' in cache_data:
            detector_data = cache_data['detector_data'][:n]
            print(f"  Detector (transformed) shape: {detector_data.shape}")
            
            # Load transform to inverse transform detector data
            transform = load_transform_from_checkpoint(run_dir)
            if transform is not None:
                detector_physical = transform.inverse_transform(detector_data)
                print(f"  Applied inverse transform to detector data")
            else:
                detector_physical = detector_data
                print("  Warning: Using detector data without inverse transform")
        else:
            print("Error: detector_data not found in cache")
            continue
        
        # Build comparison dict
        comparison = {
            'Original': truth_physical,
            'Detector': detector_physical,
            'Unfolded': samples,
        }
        
        # Detect if this is MC-POM (24 columns) or mock (1D)
        is_mcpom = samples.ndim == 2 and samples.shape[1] == 24
        
        # Determine output path
        output_path = os.path.join(run_dir, f'distributions_comparison_{suffix}.pdf')
        
        # Generate PDF plot
        print(f"Generating PDF plot...")
        if is_mcpom:
            plot_distributions_multiple_with_ratio(
                comparison,
                output_path,
                reference_label='Original',
            )
        else:
            # For non-MCPOM (e.g., 10-column reduced features), use custom column mapping
            n_cols = samples.shape[1] if samples.ndim == 2 else 1
            if n_cols > 1:
                # Create column mapping dynamically
                cols = {f'Feature {i}': i for i in range(n_cols)}
                plot_distributions_multiple_with_ratio(
                    comparison,
                    output_path,
                    cols=cols,
                    reference_label='Original',
                )
            else:
                # 1D case
                plot_distributions_1d_with_ratio(
                    comparison,
                    output_path,
                    reference_label='Original',
                )
        
        print(f"Saved PDF plot to {output_path}")


if __name__ == '__main__':
    main()
