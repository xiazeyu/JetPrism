#!/usr/bin/env python
"""Plot checkpoint evolution for specific epochs only."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt

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

import torch
from jetprism import utils
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to run directory")
    parser.add_argument("--epochs", nargs="+", type=int, required=True,
                        help="Specific epochs to include (e.g., --epochs 9 19 29 109 209)")
    parser.add_argument("--n_generate", type=int, default=100000)
    parser.add_argument("--bins", type=int, default=200)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_path = Path(args.run_dir)
    
    # Find all checkpoints
    all_checkpoints = utils.find_checkpoints(run_path)
    if not all_checkpoints:
        log.error(f"No checkpoints found in {run_path}")
        return
    
    # Filter to requested epochs
    epoch_set = set(args.epochs)
    filtered = [(e, p) for e, p in all_checkpoints if e in epoch_set]
    
    if not filtered:
        log.error(f"No checkpoints matched epochs {args.epochs}")
        log.info(f"Available epochs: {[e for e, p in all_checkpoints]}")
        return
    
    log.info(f"Plotting evolution for {len(filtered)} checkpoints: epochs {[e for e, p in filtered]}")
    
    # Locate checkpoints dir for transform
    actual_ckpt_dir = run_path
    if not list(run_path.glob("epoch_*.ckpt")) and (run_path / "checkpoints").is_dir():
        actual_ckpt_dir = run_path / "checkpoints"
    
    transform = utils.load_checkpoint_transform(actual_ckpt_dir, device)
    
    # Generate grid plot
    utils.plot_checkpoint_evolution_grid(
        checkpoints=filtered,
        n_generate=args.n_generate,
        device=device,
        output_path=str(run_path / "checkpoint_evolution_grid_custom.pdf"),
        transform=transform,
        bins=args.bins,
    )
    
    log.info(f"Saved plot to {run_path}")


if __name__ == "__main__":
    main()
