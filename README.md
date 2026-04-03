# JetPrism

JetPrism trains generative models (flow matching, diffusion (experimental)) to learn the distribution of physics events, targeting the MC-POM dataset (γp → ρ⁰p → π+π-p kinematics) and synthetic benchmarks. Both unconditional generation and detector unfolding are supported.

**Models**: CFM, DDPM (experimental)

**Framework**: PyTorch Lightning + Hydra + WandB

**Python**: ≥ 3.12

> **Paper**: [arXiv:2604.01313](https://doi.org/10.48550/arXiv.2604.01313)  
> **Software Archive**: [doi:10.5281/zenodo.19364484](https://doi.org/10.5281/zenodo.19364484)

---

## Dataset & Checkpoints

Download the dataset (MC-POM) and pre-trained checkpoints from Zenodo:
> **Link**: [https://doi.org/10.5281/zenodo.19277778](https://doi.org/10.5281/zenodo.19277778)

- **Dataset (`mc_pom_v2.parquet`)**: Required for MC-POM generation or unfolding tasks. Place it in the `data/` directory.
- **Checkpoints**: Use these to skip training your own model and directly run generation or unfolding. Extract the `denoise` and `generation` folders into the `outputs/` directory.

After downloading and placing the files, your project structure should look like this:

```text
JetPrism/
├── configs/
├── data/
│   └── mc_pom_v2.parquet
├── jetprism/
├── outputs/
│   ├── denoise/ (optional)
│   │   └── mcpom/ (optional)
│   └── generation/ (optional)
│       ├── mcpom/ (optional)
│       └── mock/ (optional)
├── scripts/
└── main.py
```

---

## Installation

JetPrism uses [uv](https://docs.astral.sh/uv/) for dependency management.

> **Note**: If you have conda activated, deactivate it first (`conda deactivate`) to avoid environment conflicts with `uv`.

```bash
# Clone the repository
git clone https://github.com/xiazeyu/JetPrism.git
cd JetPrism

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync all dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

If you are running on a Volta (SM 7.0) GPU (e.g. V100) whose support was removed from PyTorch 2.11.0+ CUDA binary builds, run the following after `uv sync`. This is also required if the GPU node in your slurm cluster uses Volta GPUs:

```bash
uv pip install torch==2.10.0 torchvision==0.25.0
```

To fully replicate the environment used in the JINST paper:

```bash
uv venv
uv pip install -r requirements-jinst.txt
source .venv/bin/activate
```

---

## Basic Usage

All runs go through `main.py`, which is a [Hydra](https://hydra.cc/) app. Config files live in the `configs/` folder and are composed from groups; results are written to `outputs/YYYY-MM-DD/HH-MM-SS_<run_id>/`.

### Train

```bash
# Train CFM on MC-POM data (default)
python main.py

# Train with a specific number of samples (default uses the entire dataset)
python main.py dataset.sample_num=1000000

# Train with a different network and hyperparameters
python main.py model.network_type=mlp model.time_embed_dim=128 model.norm=layer

# Use a predefined experiment config to train on a synthetic multi-peak dataset
python main.py +experiment=mock_gen dataset=triple_mixed

# Sweep over different hyperparameter combinations (Hydra multirun)
python main.py -m dataset.random_seed=42,43 model.time_embed_dim=64,128 model.hidden_dims=[512,512,512],[512,512,512,512],[512,512,512,512,512]

# Submit SLURM job for single train or sweep
# (set SLURM_ACCOUNT and SLURM_PARTITION env vars, or pass --account / --partition)
python slurm_submit.py --submit -- python main.py dataset.random_seed=43

python slurm_submit.py --submit -- python main.py -m +experiment=mock_gen dataset=triple_mixed,delta_0,noise_10spikes model.time_embed_dim=64,128 model.hidden_dims=[512,512,512],[512,512,512,512],[512,512,512,512,512]
```

### Predict (generation and unfolding)

Generates samples from **multiple checkpoints**: 3 intermediate epochs (~40/60/80%), plus `last` and `best`.
Accepts either a **run directory** or a **checkpoint file**:

```bash
# Using run directory (recommended)
python main.py mode=PREDICT n_generate=1000000 checkpoint_path=outputs/2026-02-17/19-16-46_abc12345

# Using specific checkpoint file
python main.py mode=PREDICT n_generate=1000000 checkpoint_path=outputs/2026-02-17/19-16-46_abc12345/checkpoints/last.ckpt

# Predict only the best checkpoint (skip intermediates and last)
python main.py mode=PREDICT n_generate=1000000 predict_best_only=true checkpoint_path=outputs/2026-02-17/19-16-46_abc12345

# Save generated samples (default: none)
python main.py mode=PREDICT save_samples=none checkpoint_path=outputs/2026-02-17/19-16-46_abc12345
python main.py mode=PREDICT save_samples=best checkpoint_path=outputs/2026-02-17/19-16-46_abc12345
python main.py mode=PREDICT save_samples=all checkpoint_path=outputs/2026-02-17/19-16-46_abc12345

# Predict unfolding checkpoints
# dataset and detector configs are auto-restored from the checkpoint
python main.py mode=PREDICT checkpoint_path=outputs/2026-02-17/19-16-46_abc12345

# Submit SLURM job for single predict
# (set SLURM_ACCOUNT and SLURM_PARTITION env vars, or pass --account / --partition)
python slurm_submit.py --submit -- python main.py mode=PREDICT n_generate=1000000 checkpoint_path=outputs/2026-02-17/19-16-46_abc12345
```

Outputs per checkpoint (e.g. `400`, `last`, `best`):
- `generated_samples_{suffix}.npz` (only when `save_samples=best` or `save_samples=all`)
- `generated_distribution_{suffix}.png`
- `distributions_diff_{suffix}.png`

### Batch predict (all runs under a sweep directory)

> **Note**: This requires SLURM. It will automatically submit a sweep of SLURM jobs for each run found in the target directory.

```bash
python main.py mode=BATCH_PREDICT runs_dir=outputs/2026-03-12
python main.py mode=BATCH_PREDICT n_generate=1000000 runs_dir=multirun/2026-02-17/19-16-46
```

---

## Config Groups

Config files live under `configs/` and are composed via [Hydra](https://hydra.cc/) config groups. Model configs and base dataset/detector/transform types are registered programmatically in `configs/configs.py`; YAML presets in the subdirectories build on those base types.

| Group | Default | Common options |
|---|---|---|
| `model` | `cfm` | `cfm`, `ddpm` |
| `dataset` | `single_mcpom` | Base: `gaussian`, `multipeak`, `highcut`, `highfreq`, `uniform`, `exponential`, `delta`, `single_mcpom`, `paired_mcpom`. <br> Presets: `gauss_standard`, `gauss_narrow`, `gauss_cutoff`, `uniform_flat`, `exponential_decay`, `twin_narrow`, `tall_flat_far`, `bimodal_asym`, `narrow_wide_overlap`, `triple_mixed`, `triple_flat_spread`, `noise_3spikes`, `noise_10spikes`, `delta_0` |
| `detector` | `null` | `identity`, `mcpom_easy`, `mcpom_easy2`, `mcpom_easy3`, `mcpom_mid`, `mcpom_hard`, `mock_easy`, `mock_mid`, `mock_hard` |
| `transform` | `default_pom` | Presets: `default_mock`, `default_pom`, `full_pom`. <br> Building blocks: `standard_scaler`, `four_particle_representation`, `dlpp_representation`, `identity` |
| `trainer` | `trainer` | `trainer` (CPU/auto), `gpu` (force CUDA) |
| `experiment` | none | `delta`, `mcpom_denoise`, `mcpom_gen`, `mcpom_gen_all_channels`, `mock_denoise`, `mock_gen` (use with `+experiment=…`) |

Override any field with `group.field=value`, e.g. `trainer.epochs=100`.

---

## SLURM

`slurm_submit.py` wraps any `python main.py …` command in a SLURM job script.

```bash
# Print script to stdout
python slurm_submit.py -- python main.py +experiment=mcpom_gen

# Submit directly
# (set SLURM_ACCOUNT and SLURM_PARTITION env vars, or pass --account / --partition)
python slurm_submit.py --submit -- python main.py +experiment=mcpom_gen

# Override SLURM resources
python slurm_submit.py --account your_account --partition your_partition --time 24:00:00 --mem 32G --submit -- python main.py +experiment=mcpom_gen

# Submit a multirun sweep
python slurm_submit.py --submit -- python main.py -m dataset.random_seed=42,43 model.time_embed_dim=64,128

# Save script to a file
python slurm_submit.py --output run.sh -- python main.py +experiment=mcpom_gen
```

Default resources: `gpu:v100:1`, 4 CPUs, 24 GB RAM, 12-hour wall time.

> **Note**: The SLURM account and partition in `slurm_submit.py` are cluster-specific. Update these values to match your HPC environment.

---

## Outputs

| Path | Contents |
|---|---|
| `outputs/YYYY-MM-DD/HH-MM-SS_<id>/` | Single run output |
| `multirun/YYYY-MM-DD/HH-MM-SS/<job_id>_<id>/` | Sweep sub-run output |
| `…/checkpoints/final_model.ckpt` | Best checkpoint (val loss) |
| `…/checkpoints/epoch_NNN.ckpt` | Periodic / first-N-epoch checkpoints |
| `…/generated_samples_{suffix}.npz` | Samples from `PREDICT` mode (suffix: `400`, `last`, `best`, ...) |
| `…/run_summary.yaml` | Config snapshot for the run |

The 8-character run ID is shared between the Hydra output directory and the WandB run, so the two are trivially linked.

---

## How to Reproduce JINST Paper Results

<details>
<summary>Click to expand reproduction steps</summary>

Pre-trained checkpoints are available on [Zenodo (doi:10.5281/zenodo.19277778)](https://doi.org/10.5281/zenodo.19277778). Download and place them under `outputs/` to skip training and go directly to inference or plotting.

### Training

```bash
# Mock generation — full sweep of all synthetic datasets
python main.py -m \
  +experiment=mock_gen \
  dataset.random_seed=42 \
  dataset=bimodal_asym,delta_0,exponential_decay,gauss_cutoff,narrow_wide_overlap,noise_3spikes,noise_10spikes,tall_flat_far,triple_flat_spread,triple_mixed,uniform_flat

# MC-POM generation
python main.py \
  +experiment=mcpom_gen \
  dataset.random_seed=42

# MC-POM denoising (detector unfolding)
# Only mcpom_easy/easy2/easy3 are used in the paper.
# mcpom_mid and mcpom_hard involve dropped events; inpainting support is
# still work in progress, so those results are not included in the paper.
python main.py -m \
  +experiment=mcpom_denoise \
  dataset.random_seed=42 \
  detector=mcpom_easy,mcpom_easy2,mcpom_easy3
```

### Inference

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

### Plotting

```bash
# Distribution comparisons (mock)
python scripts/plot_distributions_diff_pdf.py outputs/generation/mock/noise_10spikes
python scripts/plot_distributions_diff_pdf.py outputs/generation/mock/triple_mixed
python scripts/plot_distributions_diff_pdf.py outputs/generation/mock/uniform_flat
python scripts/plot_distributions_diff_pdf.py outputs/generation/mock/exponential_decay
python scripts/plot_distributions_diff_pdf.py outputs/generation/mock/bimodal_asym
python scripts/plot_distributions_diff_pdf.py outputs/generation/mock/tall_flat_far

# Distribution comparisons (MC-POM)
python scripts/plot_distributions_diff_pdf.py outputs/generation/mcpom/mcpom_gen
python scripts/plot_t_closeup_from_npz.py outputs/generation/mcpom/mcpom_gen

# MC-POM denoising comparison
python scripts/plot_denoise_comparison_pdf.py \
  outputs/denoise/mcpom/sigma_1.0 \
  outputs/denoise/mcpom/sigma_0.5 \
  outputs/denoise/mcpom/sigma_2.0

# Training evolution
python scripts/plot_checkpoint_evolution_custom.py outputs/generation/mock/noise_10spikes \
  --epochs 9 19 29 39 49 59 69 79 109 209 309 409 509 609 709
python scripts/plot_loss_vs_physics_metrics.py outputs/generation/mcpom/mcpom_gen
python scripts/plot_correlation_matrix_heatmap.py outputs/generation/mcpom/mcpom_gen

# Flow trajectory visualization
python scripts/generate_flow_trajectory_pdf.py outputs/generation/mock/delta_0/final_model.ckpt
python scripts/generate_flow_trajectory_pdf.py outputs/generation/mock/triple_mixed_scale_1/final_model.ckpt

# Failure modes analysis
python scripts/plot_failure_modes.py --output figures/failure_modes.pdf
```

### Data Requirements

- **MC-POM tasks**: Require `data/mc_pom_v2.parquet` (available on Zenodo)
- **Mock tasks**: Fully reproducible from config YAML + random seed (no external data needed)

</details>

---

## Key Dependencies

Core: `torch`, `lightning`, `hydra-core`, `torchdyn`, `wandb`  
Data: `numpy`, `pandas`, `fastparquet`, `vector`, `scipy`  
Physics: `particle`, `hepunits`  
Visualization: `matplotlib`, `rich`

See `pyproject.toml` for the complete dependency list and [agent.md](agent.md) for full architecture and API reference.

---

## Citation

If you use JetPrism in your research, please cite:

```bibtex
@misc{xia2026jetprismdiagnosingconvergencegenerative,
      title={JetPrism: diagnosing convergence for generative simulation and inverse problems in nuclear physics}, 
      author={Zeyu Xia and Tyler Kim and Trevor Reed and Judy Fox and Geoffrey Fox and Adam Szczepaniak},
      year={2026},
      eprint={2604.01313},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.01313}, 
}
```

See also [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
