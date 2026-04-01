#!/bin/bash
# Prepare checkpoints for public release
# Usage: ./scripts/prepare_checkpoints_release.sh

set -e

DEST="/scratch/yxn7cj/JetPrism_checkpoints"
SRC="/scratch/yxn7cj/JetPrism/outputs"

# Create destination structure
mkdir -p "$DEST/generation/mock"
mkdir -p "$DEST/generation/mcpom"
mkdir -p "$DEST/denoise/mcpom"

# Initialize temp file for run index
rm -f "$DEST/.run_index.tmp"

# Define runs to copy: SRC_DIR:DEST_DIR:NAME
declare -a RUNS=(
    # Mock generation (from 2026-03-20)
    "2026-03-20/00-31-09_hkgrumt2:generation/mock:delta_0"
    "2026-03-20/00-31-09_nbarpdzk:generation/mock:triple_mixed"
    "2026-03-21/20-45-52_yap24iem:generation/mock:triple_mixed_scale_1"
    "2026-03-20/00-31-22_nz0otpl3:generation/mock:noise_10spikes"
    # Mock generation (from 2026-03-22 - full sweep)
    # "2026-03-22/21-40-05_bkrx6brp:generation/mock:delta_0_2"
    "2026-03-22/21-40-05_1dlviacs:generation/mock:bimodal_asym"
    "2026-03-22/21-40-05_sw2p407u:generation/mock:exponential_decay"
    "2026-03-22/21-40-05_u9qmfdkx:generation/mock:gauss_cutoff"
    "2026-03-22/21-40-05_6aijfyng:generation/mock:narrow_wide_overlap"
    "2026-03-22/21-40-05_ggo2lzyf:generation/mock:noise_3spikes"
    # "2026-03-22/21-40-05_c8opna4i:generation/mock:noise_10spikes_2"
    "2026-03-22/21-40-05_15aedvbl:generation/mock:tall_flat_far"
    "2026-03-22/21-40-05_rrzmwkhu:generation/mock:triple_flat_spread"
    # "2026-03-22/21-40-05_54jfcj9a:generation/mock:triple_mixed_2"
    "2026-03-22/21-40-05_50am7k4i:generation/mock:uniform_flat"
    # MCPOM generation
    "2026-03-20/00-31-10_eujlglkx:generation/mcpom:mcpom_gen"
    # MCPOM denoise
    "2026-03-22/04-15-05_drr3p7zs:denoise/mcpom:sigma_0.5"
    "2026-03-22/04-15-03_60ido4bi:denoise/mcpom:sigma_1.0"
    "2026-03-22/04-15-03_2ah1x9hh:denoise/mcpom:sigma_2.0"
)

# Function to select checkpoints: first, 30%, 60%, 90%, last
select_checkpoints() {
    local ckpt_dir="$1"
    local dest_dir="$2"
    
    mkdir -p "$dest_dir"
    
    # Get sorted list of epoch checkpoints
    local epochs=($(ls "$ckpt_dir"/epoch_*.ckpt 2>/dev/null | sort -t_ -k2 -n))
    local count=${#epochs[@]}
    
    if [ $count -eq 0 ]; then
        echo "  No epoch checkpoints found in $ckpt_dir"
        return
    fi
    
    # Calculate indices (0-indexed)
    local first=0
    local p30=$(( count * 30 / 100 ))
    local p60=$(( count * 60 / 100 ))
    local p90=$(( count * 90 / 100 ))
    local last=$(( count - 1 ))
    
    # Copy selected checkpoints
    echo "  Selecting checkpoints from $count total: indices $first, $p30, $p60, $p90, $last"
    cp -L "${epochs[$first]}" "$dest_dir/"
    cp -L "${epochs[$p30]}" "$dest_dir/"
    cp -L "${epochs[$p60]}" "$dest_dir/"
    cp -L "${epochs[$p90]}" "$dest_dir/"
    cp -L "${epochs[$last]}" "$dest_dir/"
    
    # Copy best.ckpt and last.ckpt if they exist
    [ -f "$ckpt_dir/best.ckpt" ] && cp -L "$ckpt_dir/best.ckpt" "$dest_dir/"
    [ -f "$ckpt_dir/last.ckpt" ] && cp -L "$ckpt_dir/last.ckpt" "$dest_dir/"
}

# Function to sanitize wandb-metadata.json
sanitize_wandb_metadata() {
    local file="$1"
    if [ -f "$file" ]; then
        python3 << EOF
import json

with open("$file", "r") as f:
    data = json.load(f)

# Fields to remove/sanitize
data["email"] = "<REMOVED>"
data["host"] = "<REMOVED>"
data["writer_id"] = "<REMOVED>"

if "git" in data:
    data["git"] = "<REMOVED>"

if "slurm" in data:
    slurm = data["slurm"]
    for key in ["job_uid", "job_gid", "job_account", "job_user", "submit_host", "host", "job_partition", "jobid"]:
        if key in slurm:
            slurm[key] = "<REMOVED>"
    # Also sanitize nodelist and other potentially identifying info
    for key in ["job_nodelist", "nodelist", "topology_addr", "submit_dir"]:
        if key in slurm:
            slurm[key] = "<REMOVED>"

with open("$file", "w") as f:
    json.dump(data, f, indent=2)

print(f"  Sanitized: $file")
EOF
    fi
}

# Function to sanitize run_summary.yaml
sanitize_run_summary() {
    local file="$1"
    if [ -f "$file" ]; then
        # Remove or genericize paths and wandb URLs
        sed -i 's|/sfs/weka/scratch/[^/]*/|/path/to/|g' "$file"
        sed -i 's|/scratch/[^/]*/|/path/to/|g' "$file"
        sed -i 's|wandb.ai/[^/]*/|wandb.ai/<user>/|g' "$file"
        echo "  Sanitized: $file"
    fi
}

# Process each run
for run_spec in "${RUNS[@]}"; do
    IFS=':' read -r src_path dest_cat name <<< "$run_spec"
    
    src_dir="$SRC/$src_path"
    dest_dir="$DEST/$dest_cat/$name"
    
    echo "Processing: $name"
    echo "  Source: $src_dir"
    echo "  Dest: $dest_dir"
    
    if [ ! -d "$src_dir" ]; then
        echo "  ERROR: Source directory not found!"
        continue
    fi
    
    # Create destination
    mkdir -p "$dest_dir"
    
    # Copy main files (dereference symlinks with -L)
    cp -L "$src_dir/final_model.ckpt" "$dest_dir/" 2>/dev/null || echo "  Warning: final_model.ckpt not found"
    cp -L "$src_dir/run_summary.yaml" "$dest_dir/" 2>/dev/null || echo "  Warning: run_summary.yaml not found"
    cp -L "$src_dir/main.log" "$dest_dir/" 2>/dev/null || echo "  Warning: main.log not found"
    
    # Copy .hydra config
    if [ -d "$src_dir/.hydra" ]; then
        cp -rL "$src_dir/.hydra" "$dest_dir/"
    fi
    
    # Copy selected checkpoints
    if [ -d "$src_dir/checkpoints" ]; then
        select_checkpoints "$src_dir/checkpoints" "$dest_dir/checkpoints"
    fi
    
    # Copy wandb files (keep original run-xxx folder name for tracking)
    wandb_src="$src_dir/wandb/latest-run"
    if [ -L "$wandb_src" ] || [ -d "$wandb_src" ]; then
        # Resolve symlink to actual run directory
        real_wandb=$(readlink -f "$wandb_src")
        run_id=$(basename "$real_wandb")
        
        # Copy wandb files with original run-xxx folder name
        mkdir -p "$dest_dir/wandb/$run_id"
        cp -rL "$real_wandb/files" "$dest_dir/wandb/$run_id/" 2>/dev/null || true
        cp -L "$real_wandb"/run-*.wandb "$dest_dir/wandb/$run_id/" 2>/dev/null || true
        
        # Sanitize wandb-metadata.json
        sanitize_wandb_metadata "$dest_dir/wandb/$run_id/files/wandb-metadata.json"
        
        # Track for index file
        echo "$dest_cat/$name|$run_id" >> "$DEST/.run_index.tmp"
    fi
    
    # Sanitize run_summary.yaml
    sanitize_run_summary "$dest_dir/run_summary.yaml"
    
    # Sanitize main.log (remove absolute paths)
    if [ -f "$dest_dir/main.log" ]; then
        sed -i 's|/sfs/weka/scratch/[^/]*/|/path/to/|g' "$dest_dir/main.log"
        sed -i 's|/scratch/[^/]*/|/path/to/|g' "$dest_dir/main.log"
    fi
    
    echo "  Done!"
    echo ""
done

echo "========================================="
echo "Checkpoint preparation complete!"
echo "Destination: $DEST"
echo ""

# Generate README.md with run index
cat > "$DEST/README.md" << 'HEADER'
# JetPrism Checkpoints

Pre-trained model checkpoints for the JetPrism flow-matching generative model.

---

## Downloading Pre-trained Checkpoints

Download the checkpoint repository and place it under the `outputs/` directory of your JetPrism workspace.

After setup, your directory structure should look like:

```
JetPrism/
├── outputs/
│   ├── denoise/
│   │   └── mcpom/   # MC-POM detector denoising/unfolding
│   └── generation/
│       ├── mock/    # Synthetic dataset generation tasks
│       └── mcpom/   # MC-POM physics dataset generation
├── main.py
├── configs/
├── jetprism/
└── scripts/
```

## Run Index

Maps checkpoint directories to their WandB run IDs for metric tracking.

| Checkpoint Path | WandB Run ID | Task |
|-----------------|--------------|------|
HEADER

# Add run mappings from temp file
if [ -f "$DEST/.run_index.tmp" ]; then
    while IFS='|' read -r path run_id; do
        # Extract task description from path
        task=$(echo "$path" | sed 's|/| → |g')
        echo "| \`$path\` | \`$run_id\` | $task |" >> "$DEST/README.md"
    done < "$DEST/.run_index.tmp"
    rm "$DEST/.run_index.tmp"
fi

cat >> "$DEST/README.md" << 'FOOTER'

## File Contents

Each checkpoint directory contains:
- `final_model.ckpt` — Best model weights
- `.hydra/` — Full Hydra configuration
- `run_summary.yaml` — Run metadata and final metrics
- `main.log` — Training log
- `checkpoints/` — Intermediate checkpoints (first, 30%, 60%, 90%, last, best)
- `wandb/run-xxx/` — WandB logs and metrics

---

## Training Commands

### Mock Generation (Synthetic Datasets)

```bash
# Mock generation — full sweep of all synthetic datasets
python main.py -m \
  +experiment=mock_gen \
  dataset.random_seed=42 \
  dataset=bimodal_asym,delta_0,exponential_decay,gauss_cutoff,narrow_wide_overlap,noise_3spikes,noise_10spikes,tall_flat_far,triple_flat_spread,triple_mixed,uniform_flat
```

### MC-POM Generation (Physics Dataset)

```bash
python main.py \
  +experiment=mcpom_gen \
  dataset.random_seed=42
```

### MC-POM Denoising (Detector Unfolding)

```bash
# MC-POM denoising (detector unfolding)
# Only mcpom_easy/easy2/easy3 are used in the paper.
# mcpom_mid and mcpom_hard involve dropped events; inpainting support is
# still work in progress, so those results are not included in the paper.
python main.py -m \
  +experiment=mcpom_denoise \
  dataset.random_seed=42 \
  detector=mcpom_easy,mcpom_easy2,mcpom_easy3
```

---

## Inference Commands

> **Note:** These commands assume you have already placed the checkpoints under `outputs/` as described in [Downloading Pre-trained Checkpoints](#downloading-pre-trained-checkpoints).

### Generate Samples from Checkpoint

```bash
# Mock generation

python main.py mode=PREDICT checkpoint_path=outputs/generation/mock/bimodal_asym
python main.py mode=PREDICT checkpoint_path=outputs/generation/mock/delta_0
python main.py mode=PREDICT checkpoint_path=outputs/generation/mock/exponential_decay
python main.py mode=PREDICT checkpoint_path=outputs/generation/mock/gauss_cutoff
python main.py mode=PREDICT checkpoint_path=outputs/generation/mock/narrow_wide_overlap
python main.py mode=PREDICT checkpoint_path=outputs/generation/mock/noise_3spikes
python main.py mode=PREDICT checkpoint_path=outputs/generation/mock/noise_10spikes
python main.py mode=PREDICT checkpoint_path=outputs/generation/mock/tall_flat_far
python main.py mode=PREDICT checkpoint_path=outputs/generation/mock/triple_flat_spread
python main.py mode=PREDICT checkpoint_path=outputs/generation/mock/triple_mixed
python main.py mode=PREDICT checkpoint_path=outputs/generation/mock/uniform_flat

# or slurm batch submission
# python main.py mode=BATCH_PREDICT runs_dir=outputs/generation/mock


# MC-POM generation
python main.py mode=PREDICT checkpoint_path=outputs/generation/mcpom/mcpom_gen/final_model.ckpt

# MC-POM denoising (detector unfolding)
# dataset and detector configs are auto-restored from the checkpoint
python main.py mode=PREDICT checkpoint_path=outputs/denoise/mcpom/sigma_0.5/final_model.ckpt
python main.py mode=PREDICT checkpoint_path=outputs/denoise/mcpom/sigma_1.0/final_model.ckpt
python main.py mode=PREDICT checkpoint_path=outputs/denoise/mcpom/sigma_2.0/final_model.ckpt
```

---

## Plotting Commands

### Distribution Comparisons

```bash
# Mock generation tasks
python scripts/plot_distributions_diff_pdf.py outputs/generation/mock/noise_10spikes
python scripts/plot_distributions_diff_pdf.py outputs/generation/mock/triple_mixed
python scripts/plot_distributions_diff_pdf.py outputs/generation/mock/uniform_flat
python scripts/plot_distributions_diff_pdf.py outputs/generation/mock/exponential_decay
python scripts/plot_distributions_diff_pdf.py outputs/generation/mock/bimodal_asym
python scripts/plot_distributions_diff_pdf.py outputs/generation/mock/tall_flat_far

# MC-POM generation
python scripts/plot_distributions_diff_pdf.py outputs/generation/mcpom/mcpom_gen
python scripts/plot_t_closeup_from_npz.py outputs/generation/mcpom/mcpom_gen
```

### MC-POM Denoising Comparison

```bash
python scripts/plot_denoise_comparison_pdf.py \
  outputs/denoise/mcpom/sigma_1.0 \
  outputs/denoise/mcpom/sigma_0.5 \
  outputs/denoise/mcpom/sigma_2.0
```

### Training Evolution

```bash
# Checkpoint evolution plot
python scripts/plot_checkpoint_evolution_custom.py outputs/generation/mock/noise_10spikes \
  --epochs 9 19 29 39 49 59 69 79 109 209 309 409 509 609 709

# Loss vs physics metrics
python scripts/plot_loss_vs_physics_metrics.py outputs/generation/mcpom/mcpom_gen

# Correlation matrix heatmap
python scripts/plot_correlation_matrix_heatmap.py outputs/generation/mcpom/mcpom_gen
```

### Flow Trajectory Visualization

```bash
python scripts/generate_flow_trajectory_pdf.py outputs/generation/mock/delta_0/final_model.ckpt
python scripts/generate_flow_trajectory_pdf.py outputs/generation/mock/triple_mixed_scale_1/final_model.ckpt
```

### Failure Modes Analysis

```bash
python scripts/plot_failure_modes.py --output figures/failure_modes.pdf
```

---

## Data Requirements

- **MC-POM tasks**: Require `data/mc_pom_v2.parquet` (physics dataset)
- **Mock tasks**: Fully reproducible from config YAML + random seed (no external data needed)

## Reproducibility Notes

All samples can be regenerated from checkpoints. The `dataset_cache.npz` files are NOT included to save space
since they can be reproduced from:
- Mock datasets: Config YAML defines distribution parameters + `random_seed`
- MC-POM: Sampling from `mc_pom_v2.parquet` with specified `random_seed`

Transform parameters are embedded in each checkpoint (`transform_state` key).

FOOTER

echo "Generated: $DEST/README.md"
echo ""
echo "Directory structure:"
find "$DEST" -type d | head -20
echo ""
echo "Total size:"
du -sh "$DEST"
