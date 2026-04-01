#!/usr/bin/env python
"""
Generate 2D correlation matrix heatmaps comparing truth vs. generated distributions.

Creates:
1. Side-by-side correlation matrices (truth vs generated)
2. Difference heatmap showing correlation discrepancies
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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


# Column renaming: raw data column names -> LaTeX physics labels
COLUMN_RENAME = {
    # Derived Observables
    't': r'$t$',
    'mpipi': r'$M_{\pi\pi}$',
    'costh': r'$\cos\theta$',
    'phi': r'$\phi$',
    # Incident Photon (γ) - stored as q0-q3
    'q0': r'$E_{\gamma}$',
    'q1': r'$p_{\gamma x}$',
    'q2': r'$p_{\gamma y}$',
    'q3': r'$p_{\gamma z}$',
    # Target Proton (p₁) - stored as p10-p13
    'p10': r'$E_{p_1}$',
    'p11': r'$p_{1x}$',
    'p12': r'$p_{1y}$',
    'p13': r'$p_{1z}$',
    # Positive Pion (π⁺) - stored as k10-k13
    'k10': r'$E_{\pi^+}$',
    'k11': r'$p_{\pi^+ x}$',
    'k12': r'$p_{\pi^+ y}$',
    'k13': r'$p_{\pi^+ z}$',
    # Negative Pion (π⁻) - stored as k20-k23
    'k20': r'$E_{\pi^-}$',
    'k21': r'$p_{\pi^- x}$',
    'k22': r'$p_{\pi^- y}$',
    'k23': r'$p_{\pi^- z}$',
    # Recoil Proton (p₂) - stored as p20-p23
    'p20': r'$E_{p_2}$',
    'p21': r'$p_{2x}$',
    'p22': r'$p_{2y}$',
    'p23': r'$p_{2z}$',
}

# Delta function columns to exclude (using raw data column names)
# photon_y=q2, target_proton_y=p12, recoil_proton_x=p21, recoil_proton_y=p22
DELTA_COLUMNS = {'q2', 'p12', 'p21', 'p22'}


def filter_and_rename_columns(data: np.ndarray, columns: list[str]) -> tuple[np.ndarray, list[str]]:
    """Remove delta function columns and rename remaining columns."""
    # Find indices to keep
    keep_idx = [i for i, c in enumerate(columns) if c not in DELTA_COLUMNS]
    
    # Filter data
    filtered_data = data[:, keep_idx]
    
    # Get filtered and renamed columns
    filtered_columns = [columns[i] for i in keep_idx]
    renamed_columns = [COLUMN_RENAME.get(c, c) for c in filtered_columns]
    
    removed = [c for c in columns if c in DELTA_COLUMNS]
    if removed:
        print(f"Removed delta function columns: {removed}")
    
    return filtered_data, renamed_columns


def load_data(output_dir: Path):
    """Load truth and generated data from output directory."""
    truth_file = output_dir / "dataset_cache.npz"
    gen_file = output_dir / "generated_samples_best.npz"
    
    print(f"Loading truth data from {truth_file}...")
    truth_data = np.load(truth_file, allow_pickle=True)
    
    print(f"Loading generated data from {gen_file}...")
    gen_data = np.load(gen_file)
    
    columns = list(truth_data['columns'])
    truth_samples = truth_data['pre_transform_data']
    gen_samples = gen_data['samples']
    
    print(f"Truth shape: {truth_samples.shape}")
    print(f"Generated shape: {gen_samples.shape}")
    print(f"Columns: {columns}")
    
    return truth_samples, gen_samples, columns


def compute_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation matrix, handling constant columns."""
    corr = np.corrcoef(data, rowvar=False)
    # Replace NaN values (from constant columns) with 0
    corr = np.nan_to_num(corr, nan=0.0)
    return corr


def plot_correlation_heatmaps(
    truth_corr: np.ndarray,
    gen_corr: np.ndarray,
    columns: list[str],
    output_path: Path,
    title_prefix: str = ""
):
    """Create side-by-side correlation heatmaps and difference plot."""
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Common settings
    vmin, vmax = -1, 1
    cmap = 'RdBu_r'
    
    # Truth correlation matrix
    im0 = axes[0].imshow(truth_corr, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    axes[0].set_title(f'{title_prefix}Truth Correlation Matrix', fontweight='bold')
    axes[0].set_xticks(range(len(columns)))
    axes[0].set_yticks(range(len(columns)))
    axes[0].set_xticklabels(columns, rotation=45, ha='right')
    axes[0].set_yticklabels(columns)
    plt.colorbar(im0, ax=axes[0], shrink=0.8)
    
    # Generated correlation matrix
    im1 = axes[1].imshow(gen_corr, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    axes[1].set_title(f'{title_prefix}Generated Correlation Matrix', fontweight='bold')
    axes[1].set_xticks(range(len(columns)))
    axes[1].set_yticks(range(len(columns)))
    axes[1].set_xticklabels(columns, rotation=45, ha='right')
    axes[1].set_yticklabels(columns)
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
    
    # Difference (Generated - Truth)
    diff = gen_corr - truth_corr
    max_diff = np.max(np.abs(diff))
    im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-max_diff, vmax=max_diff, aspect='equal')
    axes[2].set_title(f'{title_prefix}Difference (Gen - Truth)\nMax |diff| = {max_diff:.4f}', 
                      fontweight='bold')
    axes[2].set_xticks(range(len(columns)))
    axes[2].set_yticks(range(len(columns)))
    axes[2].set_xticklabels(columns, rotation=45, ha='right')
    axes[2].set_yticklabels(columns)
    plt.colorbar(im2, ax=axes[2], shrink=0.8, label='Correlation Difference')
    
    plt.tight_layout()
    
    # Save as PDF and PNG
    fig.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    print(f"Saved: {output_path.with_suffix('.png')}")
    
    plt.close(fig)
    
    return diff


def plot_detailed_difference(
    diff: np.ndarray,
    columns: list[str],
    output_path: Path,
    title_prefix: str = ""
):
    """Create detailed difference heatmap with annotated values."""
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    max_diff = np.max(np.abs(diff))
    im = ax.imshow(diff, cmap='RdBu_r', vmin=-max_diff, vmax=max_diff, aspect='equal')
    
    # Add text annotations for significant differences
    for i in range(len(columns)):
        for j in range(len(columns)):
            val = diff[i, j]
            if abs(val) > 0.05:  # Only annotate significant differences
                color = 'white' if abs(val) > max_diff * 0.6 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       fontsize=10, color=color)
    
    ax.set_title(f'{title_prefix}Correlation Difference (Gen - Truth)\nMax |diff| = {max_diff:.4f}', 
                 fontweight='bold')
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha='right')
    ax.set_yticklabels(columns)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Difference')
    
    plt.tight_layout()
    
    # Save
    fig.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    print(f"Saved: {output_path.with_suffix('.png')}")
    
    plt.close(fig)


def compute_correlation_metrics(truth_corr: np.ndarray, gen_corr: np.ndarray) -> dict:
    """Compute metrics comparing correlation matrices."""
    diff = gen_corr - truth_corr
    
    # Get upper triangle (excluding diagonal)
    upper_idx = np.triu_indices_from(diff, k=1)
    upper_diff = diff[upper_idx]
    
    metrics = {
        'max_abs_diff': np.max(np.abs(diff)),
        'mean_abs_diff': np.mean(np.abs(upper_diff)),
        'rmse': np.sqrt(np.mean(upper_diff**2)),
        'max_positive_diff': np.max(upper_diff),
        'max_negative_diff': np.min(upper_diff),
        'n_diff_gt_0.1': np.sum(np.abs(upper_diff) > 0.1),
        'n_diff_gt_0.05': np.sum(np.abs(upper_diff) > 0.05),
        'frobenius_norm': np.linalg.norm(diff, 'fro'),
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Generate correlation matrix heatmaps')
    parser.add_argument('output_dir', type=str, help='Path to output directory')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Subsample size for faster computation (default: use all)')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    # Load data
    truth_samples, gen_samples, columns = load_data(output_dir)
    
    # Filter out delta function columns and rename
    truth_samples, display_columns = filter_and_rename_columns(truth_samples, columns)
    gen_samples, _ = filter_and_rename_columns(gen_samples, columns)
    
    # Subsample if requested
    if args.sample_size and args.sample_size < len(truth_samples):
        print(f"Subsampling to {args.sample_size} samples...")
        rng = np.random.default_rng(42)
        idx = rng.choice(len(truth_samples), size=args.sample_size, replace=False)
        truth_samples = truth_samples[idx]
        gen_samples = gen_samples[idx]
    
    # Compute correlation matrices
    print("Computing truth correlation matrix...")
    truth_corr = compute_correlation_matrix(truth_samples)
    
    print("Computing generated correlation matrix...")
    gen_corr = compute_correlation_matrix(gen_samples)
    
    # Compute metrics
    print("\n=== Correlation Matrix Comparison Metrics ===")
    metrics = compute_correlation_metrics(truth_corr, gen_corr)
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Main comparison plot
    plot_path = output_dir / "correlation_matrix_comparison"
    diff = plot_correlation_heatmaps(truth_corr, gen_corr, display_columns, plot_path)
    
    # Detailed difference plot
    diff_path = output_dir / "correlation_difference_detailed"
    plot_detailed_difference(diff, display_columns, diff_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
