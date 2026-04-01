#!/usr/bin/env python3
"""Plot loss and physics metric evolution during training."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import json

# CVD-safe color palette for training metrics
COLOR_LOSS = '#0072B2'      # Blue - training loss
COLOR_WASS = '#009E73'      # Teal/green - Wasserstein
COLOR_NFE = '#D55E00'       # Vermillion - NFE (number of function evaluations)
COLOR_COV = '#56B4E9'       # Sky Blue - Correlation distance

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


def get_run_id_from_dir(run_dir: Path) -> str:
    """Extract wandb run ID from run directory."""
    wandb_dir = run_dir / "wandb"
    if not wandb_dir.exists():
        raise FileNotFoundError(f"No wandb directory found at {wandb_dir}")
    
    run_dirs = list(wandb_dir.glob("run-*"))
    if not run_dirs:
        raise FileNotFoundError(f"No wandb run directory found in {wandb_dir}")
    
    # Extract run ID from directory name like "run-20260320_003410-eujlglkx"
    dir_name = run_dirs[0].name
    run_id = dir_name.split('-')[-1]
    return run_id


def load_wandb_history(run_dir: Path, wandb_project: str) -> pd.DataFrame:
    """Load wandb history via API."""
    import wandb
    
    run_id = get_run_id_from_dir(run_dir)
    print(f"Loading wandb history for run ID: {run_id}")
    
    api = wandb.Api()
    run = api.run(f'{wandb_project}/{run_id}')
    print(f"Run: {run.name}, State: {run.state}")
    
    # Get full history
    hist = run.history(samples=50000)
    print(f"Loaded {len(hist)} history records")
    return hist


def main():
    parser = argparse.ArgumentParser(description="Plot loss vs physics metrics")
    parser.add_argument("run_dir", type=str, help="Path to the run output directory")
    parser.add_argument("--wandb-project", type=str, default="zeyu-xia-uva/jetprism",
                        help="WandB project path (default: zeyu-xia-uva/jetprism)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    
    # Load metrics via wandb API
    hist = load_wandb_history(run_dir, args.wandb_project)
    
    # Exact column names as requested
    loss_col = 'train/loss_epoch'
    wass_col = 'val/wasserstein_mean'
    nfe_col = 'val/nfe'
    corr_col = 'val/correlation_distance'
    epoch_col = 'epoch'
    
    # Check available columns
    print(f"Available columns: {[c for c in hist.columns if 'loss' in c or 'val/' in c or c == 'epoch']}")
    
    # Extract data, dropping NaN rows
    loss_data = hist[[epoch_col, loss_col]].dropna() if loss_col in hist.columns else pd.DataFrame()
    wass_data = hist[[epoch_col, wass_col]].dropna() if wass_col in hist.columns else pd.DataFrame()
    nfe_data = hist[[epoch_col, nfe_col]].dropna() if nfe_col in hist.columns else pd.DataFrame()
    corr_data = hist[[epoch_col, corr_col]].dropna() if corr_col in hist.columns else pd.DataFrame()
    
    print(f"Loss points: {len(loss_data)}, Wasserstein points: {len(wass_data)}, NFE points: {len(nfe_data)}, Correlation points: {len(corr_data)}")
    
    # Create figure with 2 rows
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 6), dpi=150, sharex=True)
    
    # TOP PANEL: Training Loss (left) and Wasserstein (right)
    ax1 = ax_top
    if len(loss_data) > 0:
        ax1.set_ylabel(r'$\mathcal{L}_\mathrm{CFM}$', color=COLOR_LOSS)
        ax1.plot(loss_data[epoch_col], loss_data[loss_col], color=COLOR_LOSS, linestyle='-', linewidth=1.5, label=r'$\mathcal{L}_\mathrm{CFM}$', alpha=0.8)
        ax1.tick_params(axis='y', labelcolor=COLOR_LOSS)
    
    ax2 = ax1.twinx()
    if len(wass_data) > 0:
        ax2.set_ylabel(r'$W_1$', color=COLOR_WASS)
        ax2.plot(wass_data[epoch_col], wass_data[wass_col], color=COLOR_WASS, linestyle='-', linewidth=1.5, label=r'$W_1$', alpha=0.9, marker='o', markersize=3)
        ax2.tick_params(axis='y', labelcolor=COLOR_WASS)
    ax1.grid(True, alpha=0.3)
    
    # Combined legend for top panel
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # BOTTOM PANEL: NFE (left) and Correlation Distance (right)
    ax3 = ax_bot
    ax3.set_xlabel('Epoch')
    if len(nfe_data) > 0:
        ax3.set_ylabel('NFE', color=COLOR_NFE)
        ax3.plot(nfe_data[epoch_col], nfe_data[nfe_col], color=COLOR_NFE, linestyle='-', linewidth=1.5, label='NFE', alpha=0.9, marker='s', markersize=3)
        ax3.tick_params(axis='y', labelcolor=COLOR_NFE)
    
    ax4 = ax3.twinx()
    if len(corr_data) > 0:
        ax4.set_ylabel(r'$D_{\mathrm{corr}}$', color=COLOR_COV)
        ax4.plot(corr_data[epoch_col], corr_data[corr_col], color=COLOR_COV, linestyle='-', linewidth=1.5, label=r'$D_{\mathrm{corr}}$', alpha=0.9, marker='^', markersize=3)
        ax4.tick_params(axis='y', labelcolor=COLOR_COV)
    ax3.grid(True, alpha=0.3)
    
    # Combined legend for bottom panel
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper right')
    
    # Set x limits
    max_epoch = max(
        loss_data[epoch_col].max() if len(loss_data) > 0 else 0,
        nfe_data[epoch_col].max() if len(nfe_data) > 0 else 0,
    )
    ax3.set_xlim(0, max_epoch)
    
    # Add horizontal reference lines at final values
    if len(nfe_data) > 0:
        final_nfe = nfe_data[nfe_col].iloc[-1]
        ax3.axhline(y=final_nfe, color=COLOR_NFE, linestyle='--', alpha=0.3, linewidth=1)
    if len(corr_data) > 0:
        final_corr = corr_data[corr_col].iloc[-1]
        ax4.axhline(y=final_corr, color=COLOR_COV, linestyle='--', alpha=0.3, linewidth=1)

    plt.tight_layout()
    
    # Save
    output_path = run_dir / "loss_vs_physics_metrics.pdf"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")
    
    output_png = run_dir / "loss_vs_physics_metrics.png"
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_png}")
    
    plt.close()
    
    # Print final values
    print(f"\nFinal metrics:")
    if len(loss_data) > 0:
        print(f"  train/loss_epoch: {loss_data[loss_col].iloc[-1]:.6f}")
    if len(wass_data) > 0:
        print(f"  val/wasserstein_mean: {wass_data[wass_col].iloc[-1]:.6f}")
    if len(nfe_data) > 0:
        print(f"  val/nfe: {nfe_data[nfe_col].iloc[-1]:.0f}")
    if len(corr_data) > 0:
        print(f"  val/correlation_distance: {corr_data[corr_col].iloc[-1]:.6f}")


if __name__ == "__main__":
    main()
