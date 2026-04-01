#!/usr/bin/env python
"""Generate flow trajectory plots in PDF format for a checkpoint."""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

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

from pathlib import Path
from jetprism.utils import plot_flow_trajectory_for_checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate flow trajectory plots in PDF')
    parser.add_argument('checkpoint_path', type=str, help='Path to the checkpoint file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: <run_dir>/flow_trajectory)')
    parser.add_argument('--n-generate', type=int, default=100000,
                        help='Number of samples for trajectory (default: 100000)')
    parser.add_argument('--num-steps', type=int, default=200,
                        help='Number of ODE integration steps (default: 200)')
    parser.add_argument('--dims', type=int, nargs=2, default=[0, 1],
                        help='Two dimensions to visualize (default: 0 1)')
    parser.add_argument('--format', type=str, default='pdf', choices=['png', 'pdf'],
                        help='Output format (default: pdf)')
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is None:
        ckpt_path = Path(args.checkpoint_path).resolve()
        run_dir = ckpt_path.parent
        if run_dir.name == 'checkpoints':
            run_dir = run_dir.parent
        output_dir = str(run_dir / 'flow_trajectory')
    else:
        output_dir = args.output_dir

    plot_flow_trajectory_for_checkpoint(
        checkpoint_path=args.checkpoint_path,
        output_dir=output_dir,
        n_generate=args.n_generate,
        num_steps=args.num_steps,
        dims=tuple(args.dims),
        save_format=args.format,
    )
    
    # Print output paths
    print(f"\nOutput directory: {output_dir}")
    for pdf_file in sorted(Path(output_dir).glob(f'*.{args.format}')):
        print(f"  {pdf_file}")
