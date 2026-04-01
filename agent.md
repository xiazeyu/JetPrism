# JetPrism Agent Reference

Comprehensive guide for AI assistants and developers. Covers architecture, every module,
each public class/function, config system, and operational workflows.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Layout](#2-repository-layout)
3. [Entry Points](#3-entry-points)
4. [Module Reference](#4-module-reference)
   - [jetprism/schemas.py](#41-jetprism-schemaspy)
   - [jetprism/datasets.py](#42-jetprism-datasetspy)
   - [jetprism/detectors.py](#43-jetprism-detectorspy)
   - [jetprism/transforms.py](#44-jetprism-transformspy)
   - [jetprism/networks.py](#45-jetprism-networkspy)
   - [jetprism/models.py](#46-jetprism-modelspy)
   - [jetprism/kinematic.py](#47-jetprism-kinematicpy)
   - [jetprism/metric.py](#48-jetprism-metricpy)
   - [jetprism/utils.py](#49-jetprism-utilspy)
   - [jetprism/constraint.py](#410-jetprism-constraintpy)
5. [Config System](#5-config-system)
   - [Hydra + Structured Configs](#51-hydra--structured-configs)
   - [Config Groups](#52-config-groups)
   - [YAML Overrides](#53-yaml-overrides)
   - [Full Config Reference](#54-full-config-reference)
6. [Operational Modes](#6-operational-modes)
7. [Data Flow End-to-End](#7-data-flow-end-to-end)
8. [Checkpoint Format](#8-checkpoint-format)
9. [WandB Integration](#9-wandb-integration)
10. [SLURM Submission](#10-slurm-submission)
11. [Analysis Scripts (`scripts/`)](#11-analysis-scripts-scripts)
12. [Known Patterns & Conventions](#12-known-patterns--conventions)

---

## 1. Project Overview

JetPrism trains generative models (flow matching, diffusion) to learn the distribution
of physics events in the MC-POM dataset or synthetic benchmark datasets.  
Both *unconditional generation* (pure generative task) and *detector unfolding*
(paired reconstruction: detector-level → particle-level) are supported.

**Key models**: DDPM, CFM.  
**Key dataset**: MC-POM (24-column parquet of γp → ρ⁰p → π+π-p kinematics).  
**Framework**: PyTorch Lightning + Hydra config management + WandB logging.

---

## 2. Repository Layout

```
JetPrism/
├── main.py                  # CLI entry point (Hydra app)
├── slurm_submit.py          # SLURM job script generator/submitter
├── pyproject.toml           # Python deps (uv); includes pinned `jinst` dependency group
├── configs/
│   ├── configs.py           # Structured config dataclasses (ConfigStore)
│   ├── default.yaml         # Hydra defaults list
│   ├── dataset/             # Dataset YAML overrides
│   ├── detector/            # Detector YAML overrides
│   ├── experiment/          # Experiment compound configs
│   ├── trainer/             # Trainer YAML (gpu.yaml)
│   └── transform/           # Transform YAML overrides
├── scripts/
│   ├── eval_best_checkpoint.py         # Evaluate metrics on best checkpoint from pre-generated samples
│   ├── generate_flow_trajectory_pdf.py # Flow trajectory visualization
│   ├── plot_checkpoint_evolution_custom.py  # Checkpoint evolution plot for specific epochs
│   ├── plot_correlation_matrix_heatmap.py   # Correlation matrix heatmap
│   ├── plot_denoise_comparison_pdf.py       # Denoising comparison plot
│   ├── plot_distributions_diff_pdf.py       # Distribution comparison plot
│   ├── plot_failure_modes.py                # Failure mode analysis plot
│   ├── plot_loss_vs_physics_metrics.py      # Loss vs physics metrics plot
│   └── plot_t_closeup_from_npz.py           # Closeup plot of t distribution
└── jetprism/
    ├── __init__.py
    ├── schemas.py           # Mode enum, EventNumpy dataclass
    ├── datasets.py          # Dataset classes (MCPom, Synthetic hierarchy)
    ├── detectors.py         # Detector classes (smearing, cuts)
    ├── transforms.py        # Feature transform classes (scaler, log, physics reps.)
    ├── networks.py          # Neural network building blocks
    ├── models.py            # PyTorch Lightning model classes
    ├── kinematic.py         # Physics kinematic calculations
    ├── constraint.py        # Physics conservation-law checks (sanity utilities)
    ├── metric.py            # Statistical metrics (chi2, Wasserstein, KS, NN memorization, joint distribution)
    └── utils.py             # Plotting helpers and utility functions
```

---

## 3. Entry Points

### `main.py`

Hydra-decorated `main(cfg)` function.  Every run starts here.

```
python main.py [overrides...]
```

#### Private helpers (module-level)

| Function | Purpose |
|---|---|
| `_main_impl(cfg)` | Core dispatcher – reads `cfg.mode` and calls the appropriate function |
| `_write_run_summary(cfg, output_dir)` | Writes `run_summary.yaml` to the output directory |
| `_serialize_transform(transform)` | Delegates to `transform.serialize()` |
| `_deserialize_transform(state)` | Delegates to `BaseTransform.deserialize(state)` |

#### Module-level callbacks

| Class | Description |
|---|---|
| `CheckpointMetadataCallback` | Injects `model_target`, `dataset_name`, `transform_state` into every saved checkpoint |
| `EpochLoggerCallback` | Logs a progress line at the start of each epoch so SLURM logs show progress |
| `FirstNEpochsCheckpoint` | Saves a checkpoint at the end of each of the first N epochs |
| `_CachedDataset` | Lightweight stand-in for a full dataset, populated from `dataset_cache.npz`. Avoids the expensive full instantiation (parquet read + detector + transform) in PREDICT mode when the cache already exists. Exposes `paired`, `data`, `original_data`, `detector_data`, `pre_transform_data`, and `data_dim`. |

#### Public workflow functions

| Function | Mode | Description |
|---|---|---|
| `train(cfg, dataset, output_dir)` | `TRAIN` | Creates DataLoaders, instantiates model, sets up WandB logger + PL callbacks, runs `trainer.fit()`. Returns `(model, trainer)`. If `cfg.predict` is `True`, `_main_impl` automatically calls `predict()` afterwards. |
| `predict(cfg, dataset\|None, output_dir)` | `PREDICT` | Generates samples or reconstructs from detector data for **multiple checkpoints**: 3 intermediate epochs (~40/60/80% of training), plus *last* and *best* (or best only when `predict_best_only=True`). Saves per-checkpoint outputs like `generated_samples_best.npz`, `generated_distribution_last.png`, `distributions_diff_400.png`. Handles both complete and partial (failed) runs. |
| `batch_predict(cfg, output_dir)` | `BATCH_PREDICT` | Iterates all `final_model.ckpt` files under `runs_dir/*/` and calls `predict()` for each. The transient Hydra output dir is removed by `main()`. |
| `plot(cfg, dataset, output_dir)` | `PLOT` | Plots dataset distribution. For MCPom datasets, renders all 24 channels in a 6×4 panel via `plot_distributions_v0`; for other datasets, uses flattened 1-D histogram. |
| `test_flow(cfg, output_dir)` | `TEST_FLOW` | Plots ODE trajectory from a checkpoint using `utils.plot_flow_trajectory_for_checkpoint`. |
| `checkpoint_evolution(cfg, output_dir)` | `CHECKPOINT_EVOLUTION` | Plots how generated distribution evolves across saved checkpoints. |

### `slurm_submit.py`

Generates (and optionally submits) SLURM job scripts.  
See the script docstring for full usage examples.

Key function: `infer_job_name(cmd_tokens) -> str` — derives a human-readable job name
from the provided Hydra command tokens (experiment/model/dataset/mode).

---

## 4. Module Reference

### 4.1 `jetprism/schemas.py`

**`Mode(Enum)`**  
Enum of all operational modes.

| Value | String |
|---|---|
| `TRAIN` | `"train"` |
| `PREDICT` | `"predict"` |
| `BATCH_PREDICT` | `"batch_predict"` |
| `PLOT` | `"plot"` |
| `TEST_FLOW` | `"test_flow"` |
| `CHECKPOINT_EVOLUTION` | `"checkpoint_evolution"` |

**`EventNumpy`** (dataclass)  
Container for a physics event as `vector.MomentumNumpy4D` arrays.  
Fields: `q` (photon), `p1` (target proton), `p2` (recoil proton), `k1` (π+), `k2` (π−).

---

### 4.2 `jetprism/datasets.py`

All dataset classes are `@dataclass` subclasses of `BaseDataset`.

#### `BaseDataset(Dataset)`

Base class providing common fields, `__len__`, `__getitem__`, train/val/test splitting,
cache save/load, and the shared data-pipeline helper.

**Fields (all marked `MISSING` – must be set via config)**

| Field | Type | Description |
|---|---|---|
| `batch_size` | `int` | Mini-batch size for DataLoaders |
| `shuffle` | `bool` | Whether to shuffle training data |
| `num_workers` | `int` | DataLoader worker processes |
| `split_ratios` | `tuple[float, float, float]` | Train / val / test fractions.  Default `(1.0, 0.0, 0.0)` — train on the full dataset; a ≤ 50 000-event val subset is auto-sampled for distribution metrics. |
| `random_seed` | `Optional[int]` | RNG seed for reproducibility |
| `paired` | `bool` | If `True`, returns `(particle_level, detector_level)` tuples |
| `detector` | `Optional[BaseDetector]` | Detector to apply (or `None`) |
| `transform` | `Optional[BaseTransform]` | Feature transform (or `None`) |
| `data_dir` | `str` | Root data directory |

**Auto-populated fields**

| Field | Description |
|---|---|
| `data` | Transformed unpaired data array `[N, D]` |
| `original_data` | Transformed particle-level data (paired mode) |
| `detector_data` | Transformed detector-level data (paired mode) |
| `pre_transform_data` | Data *before* transform (physical units, for plots) |
| `data_dim` | Feature dimension `D` after transform |

**Key methods**

| Method | Signature | Description |
|---|---|---|
| `get_splits` | `() -> (Subset, Subset, Subset)` | Returns train/val/test subsets.  When `split_ratios = (1.0, 0.0, 0.0)` (the default — **no holdout**), returns the full dataset as the train subset and a reproducible random sample of ≤ 50 000 events as the val subset (used for chi2/KS/Wasserstein monitoring only); test subset is empty. |
| `save_data` | `(save_dir: str) -> str` | Saves all arrays to `dataset_cache.npz`; returns path |
| `load_cached_data` | `(cache_path: str)` | Loads arrays from an existing cache |
| `_setup_data_from_df` | `(df: pd.DataFrame)` | **Shared pipeline**: applies detector in batches of `_DETECTOR_BATCH_SIZE` (500 000) rows, collecting numpy arrays per-batch to avoid full-size intermediate DataFrames → stores arrays → fits+applies transform → sets `data_dim`. Called by all concrete `__post_init__` methods. |

#### `MCPom(BaseDataset)`

Loads real MC-POM data from a Parquet file.

| Field | Default | Description |
|---|---|---|
| `file_name` | MISSING | Parquet filename inside `data_dir` |
| `sample_num` | MISSING | Rows to sample (None = all) |

`columns` (fixed list of 24 strings) and `column_name` (dict to display names) are predefined.

#### `Synthetic(BaseDataset)`

Abstract base for generated datasets. Subclasses implement `generate_data() -> np.ndarray`.

| Field | Default | Description |
|---|---|---|
| `sample_num` | MISSING | Number of samples |
| `dim` | MISSING | Feature dimension |

**Concrete subclasses**

| Class | Extra fields | Distribution |
|---|---|---|
| `Gaussian` | `mean`, `std` | Isotropic Gaussian |
| `HighCut` | `mean`, `std`, `threshold`, `buffer_multiplier` | Gaussian with upper cut |
| `MultiPeak` | `peaks: list[[mean,std,weight]]` | Gaussian mixture |
| `HighFrequency` | `base_mean/std`, `noise_prob/peaks/std/range` | Gaussian + sharp spike noise |
| `Uniform` | `low`, `high` | Uniform on [low, high] |
| `Exponential` | `scale`, `loc` | Shifted exponential (loc + Exp(scale)) |
| `DeltaFunction` | `center` | Constant (degenerate) distribution |

---

### 4.3 `jetprism/detectors.py`

All detector classes are `@dataclass` subclasses of `BaseDetector`.  
They operate on `pd.DataFrame` and return a modified (or filtered) `pd.DataFrame`.

**`BaseDetector`** — abstract base with `apply(df) -> df`.

| Class | Fields | Effect |
|---|---|---|
| `Identity` | — | No-op; returns input unchanged |
| `Compose` | `detectors: list` | Applies detectors sequentially |
| `CosThetaCut` | `threshold: float` | Keeps rows where `|costh| ≤ threshold` |
| `ValueCut` | `column`, `min_value`, `max_value` | Keeps rows in `[min_value, max_value]` |
| `MomentumSmearing` | `sigma: float` | Gaussian smearing on π+/π− momenta (σ proportional to p²); recomputes 4-momentum and kinematic variables |
| `GeneralSmearing` | `sigma: float` | Gaussian smearing on all numeric columns (σ proportional to |value|) |
| `UniformPhi` | `num_bins: int`, `random_state: int\|None` | Downsamples events to flatten the φ distribution by binning in φ and sampling the minimum count per bin |

---

### 4.4 `jetprism/transforms.py`

All transform classes are `@dataclass` subclasses of `BaseTransform`.  
They work on `np.ndarray` of shape `[N, D]`.

**`BaseTransform`**

Abstract base. Key contract:

```python
fit(data: np.ndarray) -> None         # learns parameters; sets dim_input/dim_output
transform(data: np.ndarray) -> np.ndarray
inverse_transform(data: np.ndarray) -> np.ndarray
serialize() -> dict                    # for checkpoint storage
BaseTransform.deserialize(state: dict) -> Optional[BaseTransform]  # static
```

`data_dim` property returns `dim_output` (raises if not yet fitted).

| Class | Fields | Operation |
|---|---|---|
| `Identity` | — | No-op |
| `Compose` | `transforms: list` | Sequential pipeline; checks column compatibility between stages |
| `StandardScaler` | `mean`, `std`, `scale=1.0` | `output = scale * (x - mean) / std` |
| `LogTransformer` | `columns: list[int]`, `offset=1.0` | `log(x + offset)` on selected columns |
| `FourParticleRepresentation` | — | Selects 3-momentum components of all 4 particles from 24-col → 12-col |
| `ReduceRedundantv1` | — | Drops transverse momenta of photon/target-proton → 10-col |
| `DLPPRepresentation` | — | Converts 24-col → `[log(pt), η, arctanh(φ/π)]` * 4 particles |

**Serialization**:  `Compose._deserialize` reconstructs each sub-transform via the
`type_map` registry inside `BaseTransform.deserialize`.  New transform types must be
added to `type_map`.

---

### 4.5 `jetprism/networks.py`

Pure `nn.Module` building blocks, no training logic.

#### Embedding modules

| Class | Purpose |
|---|---|
| `SinusoidalEmbedding(dim, max_period)` | Fixed sinusoidal time embed (DDPM-style) |
| `LearnedEmbedding(input_dim, embed_dim)` | Legacy 2-layer MLP embed; kept for backward-compat checkpoint loading |
| `FourierEmbedding(embed_dim, max_freq)` | **Recommended** – fixed Fourier bases + learned linear projection; superior for flow matching |

#### Backbone networks

| Class | Fields | Description |
|---|---|---|
| `MLP` | `input_dim, output_dim, hidden_dims, activation, norm, dropout` | Plain feed-forward MLP |
| `ResidualNetwork` | same + `hidden_dim: list` | Stack of `ResBlock` layers with skip connections |

#### Conditioned velocity/noise predictors

| Class | Embedding | Description |
|---|---|---|
| `DiffusionMLP` | `SinusoidalEmbedding` | MLP for DDPM; input `[x ‖ t_embed]` |
| `FlowMatchingMLP` | `FourierEmbedding` (default) or `LearnedEmbedding` | MLP for CFM; input `[x_t ‖ t_embed]` |
| `FlowMatchingResNet` | `FourierEmbedding` (default) or `LearnedEmbedding` | ResNet for CFM; same interface as `FlowMatchingMLP` |
| `ConditionalMLP` | Sinusoidal or Learned | MLP with explicit conditioning vector `cond`; input `[x ‖ t_embed ‖ cond_embed]` |
| `ConditionalFlowMatchingMLP` | `FourierEmbedding` (default) or `LearnedEmbedding` | MLP for conditional CFM (denoising/unfolding); input `[x_t ‖ t_embed ‖ cond_embed]` where `cond` is detector-level data |
| `ConditionalFlowMatchingResNet` | `FourierEmbedding` (default) or `LearnedEmbedding` | ResNet for conditional CFM; same interface as `ConditionalFlowMatchingMLP` |

**`ODEWrapper(model)`** — thin `nn.Module` adapter so that a velocity network
`model(x, t_batch)` can be used with `torchdyn.NeuralODE` (which calls `forward(t, x, args)`).

**`ConditionalODEWrapper(model)`** — similar adapter for conditional velocity networks;
stores a conditioning tensor via `set_condition(cond)` and passes it through during
ODE integration.

---

### 4.6 `jetprism/models.py`

All model classes are `pl.LightningModule` subclasses.

#### Free utility functions

| Function | Signature | Description |
|---|---|---|
| `sample_conditional_pt` | `(x0, x1, t, sigma) -> x_t` | Linear interpolation with optional Gaussian jitter: `x_t = t·x1 + (1−t)·x0 + σ·ε` |
| `compute_conditional_vector_field` | `(x0, x1) -> u_t` | Returns `x1 − x0` (target velocity for linear interpolation) |

#### `BaseGenerativeModel(pl.LightningModule)`

Common training infrastructure.

**Constructor args** (all subclasses inherit)

| Arg | Default | Description |
|---|---|---|
| `data_dim` | required | Feature dimension |
| `learning_rate` | `5e-4` | AdamW learning rate |
| `weight_decay` | `1e-5` | AdamW weight decay |
| `scheduler` | `"plateau"` | LR scheduler: `"plateau"` / `"cosine"` / `"none"` |
| `scheduler_patience` | `10` | Plateau patience (epochs) |
| `scheduler_factor` | `0.5` | Plateau reduction factor |
| `grad_clip_val` | `None` | Gradient clip norm |
| `metric_sample_size` | `5000` | Samples used for distribution metrics. Overridden to `100000` by `ModelConfig` in `configs.py`, and to `1000000` in experiment YAMLs. Validation generation uses a fixed seed (0) so the metric is reproducible across epochs. |

**Key methods**

| Method | Description |
|---|---|
| `configure_optimizers` | AdamW + optional LR scheduler |
| `on_before_optimizer_step` | Logs `train/grad_norm` |
| `on_train_epoch_end` | Aggregates `train/loss_epoch` |
| `on_validation_epoch_end` | Aggregates `val/loss_epoch`; periodically calls `_compute_and_log_distribution_metrics` |
| `_compute_and_log_distribution_metrics` | Generates samples, computes per-feature chi2/KS/Wasserstein and joint distribution metrics (correlation distance, covariance distance, 2D χ²) vs. buffered val data |
| `sample(n_generate, device)` | **Abstract** – generate samples |
| `reconstruct(x0)` | **Abstract** – unfold from source |

#### `DDPM(BaseGenerativeModel)`

Denoising Diffusion Probabilistic Model (Ho et al. 2020).

| Arg | Default | Description |
|---|---|---|
| `num_timesteps` | `1000` | Diffusion steps |
| `beta_start` | `1e-4` | Noise schedule start |
| `beta_end` | `0.02` | Noise schedule end |
| `beta_schedule` | `"linear"` | `"linear"` or `"cosine"` |
| `prediction_type` | `"epsilon"` | `"epsilon"`, `"x0"`, or `"v"` |
| `network_type` | (implicit: `DiffusionMLP`) | |

**Key methods**: `q_sample`, `p_losses`, `p_sample`, `sample`.

#### `CFM(BaseGenerativeModel)`

Conditional Flow Matching (Lipman et al. ICLR 2023).

| Arg | Default | Description |
|---|---|---|
| `sigma` | `0.0` | Path noise σ |
| `solver` | `"dopri5"` | ODE solver (torchdyn) |
| `solver_atol` | `1e-5` | Absolute tolerance |
| `solver_rtol` | `1e-5` | Relative tolerance |
| `network_type` | `"mlp"` | `"mlp"` → `FlowMatchingMLP`, `"resnet"` → `FlowMatchingResNet` |
| `time_embedding` | `"fourier"` | `"fourier"` or `"learned"` (legacy) |

**Key methods**

| Method | Description |
|---|---|
| `compute_loss(x0, x1, return_stats)` | MSE(v_θ(x_t,t), x1−x0) |
| `sample(n_generate, device)` | Samples Gaussian x0, then calls `reconstruct` |
| `reconstruct(x0)` | ODE integration x0→x1 via `NeuralODE` (memory-efficient: only returns final state) |
| `get_trajectory(x0, num_steps)` | Stores full trajectory tensor [steps, batch, dim] — **memory-intensive** |

**Training step** (for both paired and unpaired data):
- Paired: `(x1, x0) = batch` (particle, detector)
- Unpaired: `x1 = batch`, `x0 = randn_like(x1)`

---

### 4.7 `jetprism/kinematic.py`

Pure functions operating on `EventNumpy`.  All return `np.ndarray`.

| Function | Formula | Description |
|---|---|---|
| `mpipi(event)` | `(k1 + k2).mass` | Di-pion invariant mass |
| `t(event)` | `(p1 − p2).mass2` | Mandelstam `t` |
| `s(event)` | `(q + p1).mass2` | Mandelstam `s` |
| `s12(event)` | `(k1 + k2).mass2` | Sub-energy of `(k1, k2)` |
| `cos_theta(event)` | `ẑ · k1̂ / |k1|` | Cosine of polar angle of π+ |
| `phi(event)` | `arctan2(k1.y, k1.x) mod 2π` | Azimuthal angle of π+ |

---

### 4.8 `jetprism/metric.py`

Statistical metric utilities.

| Symbol | Signature | Description |
|---|---|---|
| `chi2_metric` | `(expected, observed, n_bins=50) -> float` | χ² between two 1D distributions; bins on the expected range. The observed histogram is re-normalised to the expected total before computing the statistic, so out-of-range generated events do not inflate χ². |
| `extract_feature` | `(data, feature_index) -> np.ndarray` | Extracts column `[:, feature_index]` |
| `momentum_residuals` | `(orig, test) -> np.ndarray` | Element-wise residuals `orig − test` |
| `num_params` | `(model) -> int` | Counts trainable parameters |
| `Timer` | dataclass | Wall-clock timer with CUDA synchronisation |
| `nearest_neighbor_distances` | `(query, ref, query_batch_size=1000, ref_batch_size=50000, device=None) -> Tensor` | Batched PyTorch L2 NN distances from each query point to its closest point in ref.  Memory per step ≈ `query_batch_size * ref_batch_size * 4` bytes. |
| `compute_nn_memorization_metric` | `(generated, training, nn_sample_size=80000, ...) -> dict` | High-level memorization diagnostic — computes `D_gen_to_train` and `D_train_to_train` (see PREDICT section). Returns dict with `*_mean` and `*_min` keys. |
| `correlation_matrix_distance` | `(truth, generated) -> float` | Frobenius norm of the difference between Pearson correlation matrices. Measures how well the model reproduces pairwise linear correlations across all feature channels. |
| `covariance_frobenius_distance` | `(truth, generated) -> float` | Frobenius norm of the difference between covariance matrices. Captures both correlation structure and per-feature variance. |
| `chi2_2d_metric` | `(truth_x, truth_y, gen_x, gen_y, n_bins=20) -> float` | 2D binned χ² for a pair of features. Bins defined by truth range; generated histogram re-normalised. |
| `pairwise_chi2_2d` | `(truth, generated, n_bins=20) -> dict` | Computes 2D χ² for all unique feature pairs. Returns `chi2_2d_mean` and per-pair `chi2_2d_{i}_{j}` values. |
| `compute_joint_distribution_metrics` | `(truth, generated, n_bins_2d=20) -> dict` | Computes all joint distribution metrics: correlation distance, covariance distance, and pairwise 2D χ² mean. |

---

### 4.9 `jetprism/utils.py`

Plotting helpers and miscellaneous utilities.  All plot functions are
**side-effect only** — they save files and log via `log.info`.

#### Module-level constants

| Constant | Description |
|---|---|
| `_MCPOM_COLS_LIST` | Ordered list of 24 MC-POM column names |
| `_MCPOM_SCALES` | Dict mapping column name → `{'xlim': (lo, hi)}` for standard axis limits |
| `_MCPOM_COLUMNS` | List of `(index, short_name, LaTeX_label, unit)` tuples |
| `_MCPOM_GROUPS` | Dict mapping group name → list of column indices |
| `_MCPOM_COLS_NO_DELTA` | Dict `{col_name: data_index}` for the 20-column subset (excludes near-zero columns) |

#### Distribution plot functions

| Function | Inputs | Output file |
|---|---|---|
| `plot_flatten_dataset_distribution(dataset, output_dir, name)` | Dataset object or raw array; flattens to 1D | `<name>_distribution.png` |
| `plot_distributions_v0(dataset, filepath)` | Raw `[N, 24]` array | Single-dataset 6×4 panel |
| `plot_distributions_diff_1d(truth, generated, output_path, label_a, label_b)` | Two 1-D arrays | Overlay comparison histogram for non-MCPOM data |
| `plot_distributions_diff_v0(a, b, filepath, label_a, label_b)` | Two `[N, 24]` arrays | Overlay comparison 6×4 panel |
| `plot_distributions_multiple_v0(datasets_dict, filepath, *, cols, extra_text)` | `dict[label, array]`; optional column subset dict and per-column annotation | Multi-dataset grid panel (6×4 default, or adaptive grid when `cols` is given) |
| `plot_distributions_multiple_no_delta_v0(datasets_dict, filepath, extra_text)` | Wrapper — calls `plot_distributions_multiple_v0` with `_MCPOM_COLS_NO_DELTA` | 20-column subset panel |
| `plot_distributions_multiple_1d(datasets_dict, output_filepath)` | `dict[label, array]` of 1-D arrays | Multi-dataset overlay histogram for non-MCPOM data |
| `plot_generated_vs_truth(gen, truth, output_dir, bins, max_samples)` | Two arrays | Per-group PDFs saved to `generated_vs_truth/` |

#### Model / trajectory utilities

| Function | Description |
|---|---|
| `load_model_from_checkpoint(checkpoint_path, device)` | Loads model + transform from a `.ckpt` file; falls back to heuristic class detection if `model_target` key is absent |
| `plot_flow_trajectory_for_checkpoint(checkpoint_path, output_dir, ...)` | Loads model, integrates ODE trajectory, saves scatter/density/marginal plots |
| `run_checkpoint_evolution_plot(run_dir, ...)` | Iterates epoch checkpoints in `run_dir/checkpoints/`, generates samples at each, saves overlay histogram animation |

#### Statistical analysis

| Function | Description |
|---|---|
| `fit_and_compare_gaussian(samples, truth, output_dir)` | Fits Gaussians to generated samples, compares mean/std with truth, saves summary CSV and plot |

---

### 4.10 `jetprism/constraint.py`

Physics conservation-law checks operating on `EventNumpy`.  
All functions should return values close to `0` for valid events.

| Function | Returns | Check |
|---|---|---|
| `momentum_conservation(event)` | `np.float64` | Invariant mass of `(q+p1) − (p2+k1+k2)`; should be ≈ 0 |
| `energy_conservation(event)` | `np.float64` | `(E_q + E_p1) − (E_p2 + E_k1 + E_k2)`; should be ≈ 0 |
| `mass_conservation(event)` | `np.ndarray[5]` | Per-particle mass residuals vs. PDG values |
| `zero_momentum(event)` | `np.ndarray[4]` | Transverse / zero-momentum components that should vanish |

---

## 5. Config System

### 5.1 Hydra + Structured Configs

Configuration is managed by [Hydra](https://hydra.cc/) using structured config dataclasses
registered in `configs/configs.py` via `ConfigStore`.  
`default.yaml` defines the defaults list (which group option is selected by default).

At runtime, `instantiate(cfg.dataset)` / `instantiate(cfg.model)` etc. create Python
objects from the config using the `_target_` field.

### 5.2 Config Groups

| Group | Default | Purpose |
|---|---|---|
| `path` | `path` | `data_dir`, `output_dir` (Hydra-interpolated) |
| `trainer` | `trainer` | Epochs, device, logging, checkpoint intervals |
| `dataset` | *required* | Which dataset to load |
| `detector` | `null` | Which detector to apply (None = no detector) |
| `transform` | `standard_scaler` | Which feature transform |
| `model` | *required* | Which generative model |
| `experiment` | *none* | Compound overrides (see `configs/experiment/`) |

**Registered dataset configs (base)**

| Name | Class | Notes |
|---|---|---|
| `gaussian` | `Gaussian` | `mean=0.0, std=1.0, dim=1, N=8M` |
| `highcut` | `HighCut` | Gaussian with upper cut |
| `multipeak` | `MultiPeak` | Two-peak mixture |
| `highfreq` | `HighFrequency` | Gaussian + high-freq spikes |
| `uniform` | `Uniform` | Flat on [-1, 1] |
| `exponential` | `Exponential` | Shifted exponential |
| `delta` | `DeltaFunction` | All-zero delta |
| `single_mcpom` | `MCPom` | Real data, unpaired |
| `paired_mcpom` | `MCPom` | Real data, paired (paired=True) |

**Mock dataset presets** (YAML files in `configs/dataset/`)

| Config name | Base | Difficulty | Description |
|---|---|---|---|
| `gauss_standard` | `gaussian` | Easy | Standard Gaussian N(0,1) — trivial baseline |
| `gauss_narrow` | `gaussian` | Easy | Narrow Gaussian (mean=1, std=0.1) |
| `uniform_flat` | `uniform` | Easy | Flat on [-2, 2], no peaks |
| `exponential_decay` | `exponential` | Easy | Skewed one-sided (scale=1) |
| `gauss_cutoff` | `highcut` | Medium | Gaussian with sharp cutoff at 0.8 |
| `twin_narrow` | `multipeak` | Medium | 2 equal narrow peaks (std=0.3), close |
| `tall_flat_far` | `multipeak` | Medium | Tall + flat peaks, far apart |
| `bimodal_asym` | `multipeak` | Medium | 85/15 weight split — minority mode test |
| `narrow_wide_overlap` | `multipeak` | Hard | Narrow + wide peaks overlapping |
| `triple_mixed` | `multipeak` | Hard | 3 peaks, mixed widths |
| `triple_flat_spread` | `multipeak` | Hard | 3 flat peaks, wide spread (0→9) |
| `noise_3spikes` | `highfreq` | Hard | Gaussian + 3 narrow spikes (15% noise) |
| `noise_10spikes` | `highfreq` | Hard | Gaussian + 10 narrow spikes (40% noise) |
| `delta_0` | `delta` | Hard | Point mass at 0 (degenerate stress test) |

**Registered model configs**

| Name | Class |
|---|---|
| `cfm` | `CFM` |
| `ddpm` | `DDPM` |

### 5.3 YAML Overrides

Inline overrides are written as `key=value` on the CLI.  
Example dataset YAMLs in `configs/dataset/` (e.g. `gauss_narrow.yaml`) extend the registered
Hydra configs by overriding individual fields.

**`run_id` resolver**: a custom OmegaConf resolver `${run_id:}` returns the 8-character
random run ID generated at process startup.  Embedded in the Hydra output path and WandB
run ID, so the two are trivially linked.

---

### 5.4 Full Config Reference

All structured configs live in `configs/configs.py` and are registered in the Hydra
`ConfigStore`.  CLI syntax: `python main.py <group>=<name> <group>.<field>=<value>`.

#### Group: `path`

Registered names: **`path`** (default)

| Field | Type | Default | Description |
|---|---|---|---|
| `cwd` | `str` | `${hydra:runtime.cwd}` | Hydra runtime working directory |
| `output_dir` | `Optional[str]` | `${hydra:runtime.output_dir}` | Hydra output directory for this run |
| `data_dir` | `str` | `${hydra:runtime.cwd}/data` | Root data directory (override: `path.data_dir=…`) |

---

#### Group: `trainer`

Registered names: **`trainer`** (default), `gpu`

| Field | Type | Default | Choices / Notes |
|---|---|---|---|
| `device` | `str` | auto | `"cuda"` or `"cpu"`; auto-selected at import time. `name=gpu` forces `"cuda"` |
| `epochs` | `int` | `200` | Total training epochs |
| `log_interval` | `int` | `10` | Save a checkpoint and log metrics every N epochs |
| `val_interval` | `int` | `10` | Run validation every N epochs |
| `num_sanity_val_steps` | `int` | `1` | PyTorch Lightning sanity-check batches before training |
| `save_first_n_checkpoints` | `int` | `0` | Also checkpoint at epoch-end for the first N epochs (`0` = disabled) |
| `plot_checkpoint_evolution` | `bool` | `False` | Run checkpoint-evolution plots immediately after training |
| `evolution_samples` | `int` | `50000` | Samples per checkpoint for post-training evolution plot |
| `evolution_every_n` | `int` | `1` | Plot every N-th checkpoint in post-training evolution |

---

#### Group: `dataset`

***Required.*** All dataset configs inherit the base fields below.

**Shared base fields** (`DatasetConfig`)

| Field | Type | Default | Description |
|---|---|---|---|
| `batch_size` | `int` | `20000` | DataLoader batch size |
| `shuffle` | `bool` | `True` | Shuffle training split |
| `num_workers` | `int` | `4` | DataLoader worker processes |
| `random_seed` | `Optional[int]` | `42` | RNG seed for reproducible splits |
| `split_ratios` | `tuple[float,float,float]` | `(1.0, 0.0, 0.0)` | Train / val / test fractions.  `(1.0, 0.0, 0.0)` trains on **the whole dataset**; val subset for chi2 monitoring is auto-sampled (≤ 50 000 events). |
| `paired` | `bool` | `False` | `True` → returns `(particle, detector)` tuples for unfolding |
| `detector` | interpolated | `"${detector}"` | Resolved from the `detector` config group |
| `transform` | interpolated | `"${transform}"` | Resolved from the `transform` config group |
| `data_dir` | interpolated | `"${path.data_dir}"` | Resolved from `path.data_dir` |

**Registered configs**

| Name | Class | Extra fields & defaults |
|---|---|---|
| `single_mcpom` | `MCPom` | `file_name="mc_pom_v2.parquet"`, `sample_num=None` (all rows), `paired=False` |
| `paired_mcpom` | `MCPom` | same + `paired=True` |
| `gaussian` | `Gaussian` | `sample_num=8_000_000`, `dim=1`, `mean=0.0`, `std=1.0` |
| `highcut` | `HighCut` | `sample_num=8M`, `dim=1`, `mean=0.0`, `std=1.0`, `threshold=0.7`, `buffer_multiplier=2.0` |
| `multipeak` | `MultiPeak` | `sample_num=8M`, `dim=1`, `peaks=[[-1.0,0.5,0.3],[1.0,0.5,0.7]]` (list of `[mean,std,weight]`) |
| `highfreq` | `HighFrequency` | `sample_num=8M`, `dim=1`, `base_mean=0.0`, `base_std=1.0`, `noise_prob=0.1`, `num_noise_peaks=20`, `noise_std=0.05`, `noise_range=[-3.0,3.0]` |
| `delta` | `DeltaFunction` | `sample_num=8M`, `dim=1`, `center=0.0` |

---

#### Group: `detector`

Default: **`null`** (no detector applied)

Registered base configs (from `configs.py`):

| Name | Class | Fields (non-MISSING defaults) | Effect |
|---|---|---|---|
| `identity` | `Identity` | — | Pass-through; no modification |
| `compose` | `Compose` | `detectors` (**MISSING** – must supply list) | Applies detectors sequentially |
| `cos_theta_cut` | `CosThetaCut` | `threshold` (**MISSING**) | Keeps rows where `\|cosθ\| ≤ threshold` |
| `value_cut` | `ValueCut` | `column`, `min_value`, `max_value` (all **MISSING**) | Keeps rows in `[min_value, max_value]` for a named column |
| `momentum_smearing` | `MomentumSmearing` | `sigma` (**MISSING**) | Gaussian smearing on π+/π− momenta; σ ∝ p² |
| `general_smearing` | `GeneralSmearing` | `sigma` (**MISSING**) | Gaussian smearing on all numeric columns; σ ∝ \|value\| |
| `uniform_phi` | `UniformPhi` | `num_bins=50`, `random_state=42` | Downsamples events to flatten φ distribution |

**Detector difficulty presets** (YAML files in `configs/detector/`)

| Difficulty | MCPOM Config | MCPOM Pipeline | Mock Config | Mock Pipeline |
|---|---|---|---|---|
| Easy | `mcpom_easy` | MomentumSmearing(σ=0.5) | `mock_easy` | GeneralSmearing(σ=0.1) |
| Easy2 | `mcpom_easy2` | MomentumSmearing(σ=1.0) | — | — |
| Easy3 | `mcpom_easy3` | MomentumSmearing(σ=2.0) | — | — |
| Medium | `mcpom_mid` | MomentumSmearing(σ=0.5) + CosThetaCut(0.8) | `mock_mid` | GeneralSmearing(σ=0.2) |
| Hard | `mcpom_hard` | MomentumSmearing(σ=1.0) + CosThetaCut(0.7) + ValueCut(k10≥0.3) | `mock_hard` | GeneralSmearing(σ=0.3) + ValueCut(col='0'≥0.5) |

---

#### Group: `transform`

Default: **`identity`**

| Name | Class | Fields & defaults | Effect |
|---|---|---|---|
| `identity` | `Identity` | — | Pass-through; data unchanged |
| `compose` | `Compose` | `transforms` (**MISSING** – must supply list) | Sequential pipeline of sub-transforms |
| `standard_scaler` | `StandardScaler` | `mean=None`, `std=None`, `scale=1.0` | `output = scale * (x − mean) / std`; fits mean/std from data if `None` |
| `log_transformer` | `LogTransformer` | `columns=None` (all), `offset=1.0` | `log(x + offset)` on selected column indices |
| `four_particle_representation` | `FourParticleRepresentation` | — | Selects 3-momentum of all 4 particles from 24-col DataFrame → 12-col array |
| `dlpp_representation` | `DLPPRepresentation` | — | Converts 24-col → `[log(pt), η, arctanh(φ/π)]` * 4 particles (12-col) |

**Transform YAML presets** (in `configs/transform/`)

| Config name | Pipeline | Description |
|---|---|---|
| `default_pom` | `FourParticleRepresentation` → `ReduceRedundantv1` → `StandardScaler(scale=5.0)` | Default for MC-POM datasets (24-col → 10-col, scaled) |
| `default_mock` | `StandardScaler(scale=5.0)` | Default for synthetic datasets |
| `full_pom` | See YAML | Full MC-POM transform pipeline |

---

#### Group: `model`

***Required.*** All model configs inherit the base fields below.

**Shared base fields** (`ModelConfig`)

| Field | Type | Default | Choices / Notes |
|---|---|---|---|
| `data_dim` | `Optional[int]` | `None` | Auto-inferred from dataset at runtime |
| `learning_rate` | `float` | `3e-4` | AdamW learning rate |
| `weight_decay` | `float` | `1e-5` | AdamW weight decay |
| `scheduler` | `str` | `"plateau"` | `"plateau"`, `"cosine"`, or `"none"` |
| `scheduler_patience` | `int` | `5` | ReduceLROnPlateau patience (epochs) |
| `scheduler_factor` | `float` | `0.5` | ReduceLROnPlateau reduction factor |
| `grad_clip_val` | `float\|None` | `None` | Gradient norm clip value; `None` = disabled |
| `metric_sample_size` | `int` | `100000` | Events used for chi2/KS/Wasserstein at validation. Overridden to `1000000` by experiment YAMLs. |

**`cfm`** — Conditional Flow Matching

| Field | Type | Default | Choices / Notes |
|---|---|---|---|
| `hidden_dims` | `list[int]` | `[256, 256, 256]` | Hidden layer sizes of velocity network |
| `time_embed_dim` | `int` | `64` | Dimensionality of time embedding |
| `sigma` | `float` | `0.0` | Gaussian path noise σ |
| `solver` | `str` | `"dopri5"` | torchdyn ODE solver tag; common: `"dopri5"`, `"euler"`, `"midpoint"` |
| `solver_atol` | `float` | `1e-5` | ODE absolute tolerance |
| `solver_rtol` | `float` | `1e-5` | ODE relative tolerance |
| `solver_steps` | `int` | `100` | Fixed steps for `euler`/`midpoint`/`rk4` solvers |
| `network_type` | `str` | `"mlp"` | `"mlp"` (`FlowMatchingMLP`) or `"resnet"` (`FlowMatchingResNet`) |
| `activation` | `str` | `"silu"` | `"relu"`, `"silu"`, `"gelu"`, or `"tanh"` |
| `norm` | `Optional[str]` | `None` | `"layer"`, `"batch"`, or `None` |
| `dropout` | `float` | `0.0` | Dropout probability |
| `conditional` | `bool` | `False` | If `True`, use conditional mode for denoising/unfolding |
| `cond_dim` | `int\|None` | `None` | Conditioning dimension (defaults to `data_dim`) |
| `cond_embed_dim` | `int\|None` | `None` | Embedding dimension for conditioning |

**`ddpm`** — Denoising Diffusion Probabilistic Model

| Field | Type | Default | Choices / Notes |
|---|---|---|---|
| `hidden_dims` | `list[int]` | `[256, 256, 256]` | Hidden layer sizes of noise network |
| `time_embed_dim` | `int` | `64` | Sinusoidal time embedding dimension |
| `num_timesteps` | `int` | `1000` | Diffusion steps T |
| `beta_start` | `float` | `1e-4` | Noise schedule β₀ |
| `beta_end` | `float` | `0.02` | Noise schedule β_T |
| `beta_schedule` | `str` | `"linear"` | `"linear"` or `"cosine"` |
| `prediction_type` | `str` | `"epsilon"` | `"epsilon"` (noise), `"x0"` (clean data), or `"v"` (velocity) |
| `activation` | `str` | `"silu"` | `"relu"`, `"silu"`, `"gelu"`, `"tanh"` |
| `norm` | `Optional[str]` | `None` | `"layer"`, `"batch"`, or `None` |
| `dropout` | `float` | `0.0` | Dropout probability |

---

#### Root `Config` fields

These are set directly at the CLI top-level (no group prefix).

| Field | Type | Default | Description |
|---|---|---|---|
| `mode` | `schemas.Mode` | **MISSING** | Operational mode: `TRAIN`, `PREDICT`, `BATCH_PREDICT`, `PLOT`, `TEST_FLOW`, `CHECKPOINT_EVOLUTION` |
| `predict` | `bool` | `False` | When `True` and `mode=TRAIN`, automatically run prediction after training completes |
| `checkpoint_path` | `Optional[str]` | `None` | Path to a run directory or `.ckpt` file (required for PREDICT / TEST_FLOW) |
| `n_generate` | `int` | `8000000` | Samples to generate in PREDICT mode |
| `generation_batch_size` | `int` | `20000` | Per-batch generation size (controls VRAM during ODE integration) |
| `fit_gaussian` | `bool` | `False` | Fit Gaussians to generated samples and compare with truth |
| `save_samples` | `str` | `"none"` | Save generated samples `.npz` files: `"none"` (skip all), `"best"` (best checkpoint only), `"all"` (every checkpoint) |
| `predict_best_only` | `bool` | `False` | If `True`, predict only the best checkpoint (skip intermediate epochs and last) |
| `runs_dir` | `Optional[str]` | `None` | Multirun directory for BATCH_PREDICT / CHECKPOINT_EVOLUTION |
| `flow_n_generate` | `int` | `20000` | Trajectory samples for TEST_FLOW |
| `flow_num_steps` | `int` | `100` | ODE integration steps for TEST_FLOW |
| `flow_dims` | `list` | `[0, 1]` | Dimension pair for 2-D trajectory plots |
| `flow_plot_type` | `str` | `"all"` | Plot type for TEST_FLOW: `scatter`, `density`, `marginal`, or `all` |
| `evolution_samples` | `int` | `50000` | Samples per checkpoint for CHECKPOINT_EVOLUTION |
| `evolution_bins` | `int` | `200` | Histogram bins for CHECKPOINT_EVOLUTION |
| `evolution_xmin` | `Optional[float]` | `None` | x-axis lower bound (None = auto) |
| `evolution_xmax` | `Optional[float]` | `None` | x-axis upper bound (None = auto) |
| `evolution_skip_last` | `bool` | `False` | Skip `last.ckpt` in CHECKPOINT_EVOLUTION |
| `evolution_every_n` | `int` | `1` | Plot every N-th checkpoint in CHECKPOINT_EVOLUTION |
| `compute_nn` | `bool` | `True` | Compute NN memorization metric in PREDICT mode |
| `nn_sample_size` | `int` | `80000` | Events sampled for each NN computation (≈ 1 % of 8 M mc_pom) |
| `nn_query_batch_size` | `int` | `1000` | Query batch size for NN (lower = less peak VRAM) |
| `nn_ref_batch_size` | `int` | `50000` | Reference batch size for NN (lower = less peak VRAM) |

---

## 6. Operational Modes

### TRAIN

```bash
python main.py mode=TRAIN model=cfm dataset=gaussian
python main.py mode=TRAIN model=cfm dataset=gaussian predict=true  # train + predict
python main.py -m mode=TRAIN model=cfm,ddpm dataset=gaussian,multipeak  # sweep
```

- Loads dataset, creates train/val DataLoaders.
- **No holdout by default** (`split_ratios=(1.0, 0.0, 0.0)`): the full dataset is used for
  training.  A reproducible random subsample of ≤ 50 000 events from the training set served
  as the validation set for chi2 / KS / Wasserstein monitoring.  The `val/overfit_gap` metric
  is not meaningful in this mode and should be ignored.
  To revert to a held-out validation set, set `dataset.split_ratios=[0.8,0.1,0.1]` (or
  any splits where the val fraction > 0).
- Infers `data_dim` from dataset; injects into model config.
- Saves `dataset_cache.npz` to `output_dir`.
- Checkpoints: periodic every `trainer.log_interval` epochs under `checkpoints/epoch_NNN.ckpt`;
  best model (by `val/loss_epoch`) saved as `checkpoints/best.ckpt`;
  final `checkpoints/last.ckpt` also saved.
  `final_model.ckpt` is a copy of the best checkpoint (falls back to last if no validation ran).
- WandB: logged with run ID = `_RUN_ID` (see resolver above).
- Optionally runs `checkpoint_evolution` after training if `trainer.plot_checkpoint_evolution=true`.
- Optionally runs `predict` after training if `predict=true`, generating samples and
  evaluation plots in the same output directory without a separate PREDICT run.

### PREDICT

```bash
# Using a run directory (recommended)
python main.py mode=PREDICT \
    checkpoint_path=outputs/2026-02-22/09-16-51_abc12345 \
    n_generate=8000000 generation_batch_size=20000

# Using a specific checkpoint file (also supported)
python main.py mode=PREDICT \
    checkpoint_path=outputs/2026-02-22/09-16-51_abc12345/checkpoints/best.ckpt \
    n_generate=8000000
```

- **Accepts both run directories and checkpoint files**: pass either
  `outputs/.../15-51-01_abc123/` (run dir) or `.../checkpoints/best.ckpt` (checkpoint).
  The run directory is auto-resolved in both cases.
- **Multi-checkpoint prediction**: automatically evaluates 3 intermediate epoch
  checkpoints (~40%, ~60%, ~80% of training) plus the `last.ckpt` and `best.ckpt`.
  For a 1000-epoch run this typically means epochs ~400, ~600, ~800, plus last and best.
- **Handles failed/partial runs**: if `last.ckpt` is missing (e.g. training crashed),
  the latest available `epoch_*.ckpt` is used as "last".
- Output files are **suffixed** with checkpoint identifiers:
  - `generated_samples_{suffix}.npz` — raw samples
  - `generated_distribution_{suffix}.png` — MC-POM 6×4 panel
  - `distributions_diff_{suffix}.png` — truth vs generated comparison
  - `distributions_comparison_{suffix}.png` — three-way (original/detector/generated) for denoising
  - Suffix examples: `400`, `600`, `800`, `last`, `best`
- All outputs are written **directly into the checkpoint's run directory** (i.e. the same
  folder created during training). Hydra would normally create a fresh `outputs/<date>/…`
  directory, but `main()` removes it in a `finally` block.
- Generates `n_generate` samples from `N(0,I)` via ODE integration.
- For paired (`dataset.paired=True`) runs paired reconstruction from detector data.
- **Cache-first loading**: for paired datasets, `_main_impl` checks for `dataset_cache.npz`
  in the run directory. If found, it loads a lightweight `_CachedDataset` from cache instead of
  running the full instantiation pipeline (parquet read → detector simulation → transform).
  If no cache exists, the full dataset is instantiated and the cache is saved for future runs.
- **Batched detector processing**: `BaseDataset._apply_detector_batched()` processes the
  detector in chunks of 500 000 rows (configurable via `_DETECTOR_BATCH_SIZE`).  This mirrors
  how the DataLoader feeds mini-batches during training and keeps peak memory manageable for
  detectors that create heavy intermediates (e.g. `MomentumSmearing` with `vector.array`
  4-momentum objects and kinematic calculations).  Without batching, processing 8M rows at once
  causes a transient memory spike of ~5 GB+ that can trigger OOM on memory-constrained systems.
- **Auto-cache for unpaired generation**: for unpaired datasets (`paired=False`), when no
  `dataset_cache.npz` exists in the run directory (e.g. old training runs), `_main_impl`
  instantiates the dataset, saves the cache, then frees the dataset before calling `predict()`.
  This ensures truth comparison (diff plots) and metrics (NN memorization, joint distribution)
  work correctly even for runs that predate cache saving.
- Applies inverse transform from checkpoint.
- **Metrics** (NN memorization, joint distribution) are computed once using the *best*
  checkpoint (fallback: *last*).
- **Nearest-Neighbour memorization metric** (enabled by default via `compute_nn=true`):
  Loads the training data (`data` key) from `dataset_cache.npz` in the same transformed
  space as the model output.  Computes two NN distances using batched PyTorch `cdist`
  (no scikit-learn or faiss required) to avoid OOM on large datasets:
    - `D_gen_to_train`: `nn_sample_size` generated events → whole training set.
    - `D_train_to_train`: random `nn_sample_size` training events → *remaining* training
      events (query indices excluded to prevent self-match).
  Results are appended to `run_summary.yaml` and logged to the **training WandB run**
  as run-level summary metrics under the `nn/` prefix (see WandB Integration section).
  Override batch sizes via `nn_query_batch_size` / `nn_ref_batch_size`.
- **BatchNorm note**: models with `BatchNorm1d` layers are switched to `train()` mode
  during generation so live batch statistics are used (avoids stale running-stat widening).

### BATCH_PREDICT

```bash
python main.py mode=BATCH_PREDICT runs_dir=multirun/2026-02-22/09-16-51
```

Iterates all `final_model.ckpt` files found under `runs_dir/**/` and calls `predict()` for each.
Outputs for each model are written into its own run directory (same as `PREDICT`). The
transient Hydra output directory is removed after the run completes.

### TEST_FLOW

```bash
python main.py mode=TEST_FLOW \
    checkpoint_path=outputs/.../epoch_099.ckpt \
    flow_n_generate=20000 flow_num_steps=100 flow_dims=[0,1] flow_plot_type=all
```

### CHECKPOINT_EVOLUTION

```bash
# Single run
python main.py mode=CHECKPOINT_EVOLUTION checkpoint_path=outputs/.../epoch_099.ckpt

# Batch over multirun
python main.py mode=CHECKPOINT_EVOLUTION runs_dir=multirun/2026-02-22/09-16-51
```

---

## 7. Data Flow End-to-End

```
Parquet file / generate_data()
        │
        ▼
  pd.DataFrame (raw 24 cols or synthetic)
        │
  BaseDataset._setup_data_from_df(df)
        │
        ├── [paired=False]
        │       │  detector.apply(df)   (optional)
        │       │  → self.data (np.ndarray)
        │       │  → self.pre_transform_data = self.data.copy()
        │       │  transform.fit(data); transform.transform(data)
        │       │  → self.data overwritten (transformed)
        │
        └── [paired=True]
                │  detector.apply(df.copy())   (required)
                │  → self.original_data, self.detector_data
                │  → self.pre_transform_data = original_data.copy()
                │  transform.fit(original_data); transform.transform(both)
                │  → self.original_data, self.detector_data overwritten

DataLoader (yields torch.Tensor batches)
        │
        ▼
Model.training_step(batch)
  ├── unpaired:  x1 = batch;  x0 = randn_like(x1)
  └── paired:    x1, x0 = batch  (particle, detector)
        │
        ▼
CFM.compute_loss(x0, x1)
  → t ~ U[0,1]
  → x_t = sample_conditional_pt(x0, x1, t, sigma)
  → u_t = x1 - x0
  → v_t = net(x_t, t)
  → loss = MSE(v_t, u_t)

─── Inference ───────────────────────────────────────────────

x0 ~ N(0, I)   [or detector_data]
        │
  NeuralODE.forward(x0, t=[0,1])   (dopri5 or other)
        │
        ▼
  x1 (generated samples, in transformed space)
        │
  transform.inverse_transform(x1)
        │
        ▼
  Physical-space samples  →  saved as generated_samples.npz
```

---

## 8. Checkpoint Format

PyTorch Lightning `.ckpt` files are standard serialised Lightning checkpoints.  
JetPrism injects extra keys via `CheckpointMetadataCallback`:

| Key | Type | Description |
|---|---|---|
| `model_target` | `str` | Fully-qualified Python class path (e.g. `jetprism.models.CFM`) |
| `dataset_name` | `str` | Hydra dataset config choice name |
| `transform_state` | `dict` | Serialised transform (via `transform.serialize()`) |

These keys allow loading a checkpoint without knowing the model class or transform
in advance — `predict()` auto-detects from `model_target`.

**Note on `LearnedEmbedding`**: `LearnedEmbedding` is kept in `networks.py` solely
for PyTorch to deserialise weights from old checkpoints that trained with it.
New runs always use `FourierEmbedding` (the default). No special override logic
is needed — `load_from_checkpoint` will restore the embedding type that was saved
in `hyper_parameters` automatically.

---

## 9. WandB Integration

WandB logging is configured inside `train()`.

- **Run ID**: 8-character random string `_RUN_ID` generated once per process (module-level).
  Embedded in both the Hydra output path (`..._<RUN_ID>/`) and `WandbLogger(id=_RUN_ID)`.
- **Run name**: `<leaf_folder>_<model_name>_<dataset_name>`.
- **Artifact**: the final `last.ckpt` is uploaded as a WandB model artifact named
  `model-<RUN_ID>`.  (`final_model.ckpt` is the best-val checkpoint, not necessarily last.)
- **Tracing**: after training, `wandb_id`, `wandb_name`, and `wandb_url` are appended to
  `run_summary.yaml` in the output directory.
- **Post-predict NN logging**: `predict()` resumes the original training run (by ID) to
  attach NN memorization metrics to its summary. The `entity` and `project` are parsed
  from the stored `wandb_url` so the correct run is targeted; `resume="must"` is used to
  prevent accidentally creating a new run.

### Logged Metrics

| Metric | Granularity | x-axis | Description |
|---|---|---|---|
| `train/loss` | per step | global_step | MSE loss for each training batch |
| `train/loss_epoch` | per epoch | epoch | Mean training loss across all batches in the epoch |
| `train/velocity_pred_norm` | per step | global_step | Mean L2 magnitude of predicted velocity `v_t` (CFM only) |
| `train/velocity_target_norm` | per step | global_step | Mean L2 magnitude of target velocity `u_t = x1 - x0` (CFM only) |
| `train/velocity_cos_sim` | per step | global_step | Cosine similarity between `v_t` and `u_t`; should converge toward 1.0 (CFM only) |
| `train/grad_norm` | per step | global_step | L2 gradient norm before optimizer step |
| `val/loss` | per val step | global_step | Validation MSE loss per batch (many points per epoch) |
| `val/loss_epoch` | per epoch | epoch | Mean validation loss across all val batches |
| `val/overfit_gap` | per epoch | epoch | `val/loss_epoch − train/loss_epoch`; positive = overfitting.  **Not meaningful when `split_ratios=(1.0,0.0,0.0)` (default no-holdout)** since val data is sampled from training data. |
| `val/chi2_mean` | every val epoch | epoch | Mean chi² between generated and true distributions across features |
| `val/ks_statistic_mean` | every val epoch | epoch | Mean KS statistic across features |
| `val/wasserstein_mean` | every val epoch | epoch | Mean Wasserstein distance across features |
| `val/correlation_distance` | every val epoch | epoch | Frobenius ‖corr(truth) − corr(gen)‖; measures joint correlation structure across all channels (multi-dim data only) |
| `val/covariance_distance` | every val epoch | epoch | Frobenius ‖cov(truth) − cov(gen)‖; captures both correlation and variance structure (multi-dim data only) |
| `val/chi2_2d_mean` | every val epoch | epoch | Mean 2D binned χ² over all feature pairs; tests joint distribution quality beyond marginals (multi-dim data only) |
| `val/nfe` | every val epoch | epoch | Number of ODE function evaluations during sampling (adaptive solvers only); lower = straighter flow |
| `nn/D_gen_to_train_mean` | post-predict (summary) | — | Mean L2 distance from generated events to their nearest training neighbour |
| `nn/D_train_to_train_mean` | post-predict (summary) | — | Mean L2 distance from a training sub-sample to the rest of the training set |
| `nn/D_gen_to_train_min` | post-predict (summary) | — | Minimum of the above gen-to-train distances |
| `nn/D_train_to_train_min` | post-predict (summary) | — | Minimum of the above train-to-train distances |
| `nn/memorization_ratio` | post-predict (summary) | — | `D_gen_to_train_mean / D_train_to_train_mean`; ~1.0 = good generalisation, ≪1.0 = memorisation |

**Note on x-axis**: `*_epoch` and distribution metrics (`chi2`, `ks`, `wasserstein`) use
`epoch` as the WandB x-axis (via `wandb.define_metric`). Step-level metrics use the default
`global_step`. Distribution metrics are computed on every validation epoch — their frequency
is controlled by `trainer.val_interval` (same as `check_val_every_n_epoch` in the PL Trainer).

### Why val/loss has more data points than distribution metrics

- `val/loss` is logged inside `validation_step` — once per validation batch, every epoch
  → `n_val_batches * n_val_epochs` total points.
- `val/loss_epoch` is logged once per validation epoch → `n_val_epochs` points.
- `val/chi2_mean` etc. are logged once per validation epoch (same frequency as `val/loss_epoch`);
  they have fewer points than `val/loss` only because `val/loss` accumulates per batch.

---

## 10. SLURM Submission

`slurm_submit.py` generates a self-contained SLURM batch script from any `main.py` command
and optionally submits it via `sbatch`.  Everything after `--` is treated as the verbatim
command to embed in the script; everything before `--` controls SLURM/script options.

### Quick-start: default `mcpom_gen` run

```bash
# Generate script (printed to stdout) — inspect before submitting
python slurm_submit.py -- python main.py +experiment=mcpom_gen

# Generate AND submit immediately
python slurm_submit.py --submit -- python main.py +experiment=mcpom_gen

# Save script to a named file then submit manually
python slurm_submit.py --output run_mcpom_gen.sh -- python main.py +experiment=mcpom_gen
sbatch run_mcpom_gen.sh
```

### Layer-depth sweep

For `-m` (multirun) commands, `slurm_submit.py` expands the Cartesian product of sweep
axes and submits one `sbatch` job per combination.

`slurm_submit.py` splits sweep values only on commas that are **outside** brackets, so
list-valued overrides like `[512,512,512]` are treated as a single atomic value — not
exploded on the internal commas.

```bash
# Dry-run — print the scripts that would be submitted, without running sbatch
python slurm_submit.py --dry-run --submit -- \
    python main.py -m +experiment=mcpom_gen \
    "model.hidden_dims=[512,512,512],[512,512,512,512,512],[512,512,512,512,512,512,512],[512,512,512,512,512,512,512,512,512,512]"

# Submit all four depths
python slurm_submit.py --submit -- \
    python main.py -m +experiment=mcpom_gen \
    "model.hidden_dims=[512,512,512],[512,512,512,512,512],[512,512,512,512,512,512,512],[512,512,512,512,512,512,512,512,512,512]"
```

> **Tip**: outputs land in a shared `multirun/<date>/<time>/` directory so all four runs are
> trivially compared.  Use `plot_checkpoint_evolution_custom.py` to visualize how distributions
> evolve across checkpoints.

> **Note**: SLURM account and partition are read from `SLURM_ACCOUNT` / `SLURM_PARTITION`
> environment variables. Set them in your shell profile (e.g. `~/.zshrc`) or pass
> `--account` / `--partition` on the command line.

### Other common patterns

```bash
# Override wall-time and memory for a long run
python slurm_submit.py --time 24:00:00 --mem 32G --submit -- python main.py +experiment=mcpom_gen

# Activate a specific conda environment on the cluster
python slurm_submit.py --conda-env myenv --submit -- python main.py +experiment=mcpom_gen

# Activate a virtualenv explicitly
python slurm_submit.py --venv /path/to/.venv --submit -- python main.py +experiment=mcpom_gen

# Load extra cluster modules before running
python slurm_submit.py --module cuda/12.1 --module cudnn/8.9 --submit -- python main.py +experiment=mcpom_gen

# Single predict job
python slurm_submit.py --submit -- python main.py mode=PREDICT \
    +checkpoint_path=outputs/2026-02-24/16-08-01_lf5ww2hy/final_model.ckpt \
    n_generate=4000000

# Batch predict over an entire multirun directory
python slurm_submit.py --submit -- python main.py mode=BATCH_PREDICT \
    +runs_dir=multirun/2026-02-24/16-00-00 n_generate=4000000
```

### CLI reference

All flags come **before** `--`; the verbatim `main.py` invocation comes **after** `--`.

| Flag | Default | Description |
|---|---|---|
| `--account` | `YOUR_ACCOUNT` | SLURM account |
| `--partition` | `YOUR_PARTITION` | SLURM partition |
| `--gres` | `gpu:v100:1` | Generic resource (GPU spec) |
| `--nodes` | `1` | Number of nodes |
| `--cpus` | `4` | CPUs per task |
| `--mem` | `24G` | Memory per node |
| `--time` | `12:00:00` | Wall-clock time limit |
| `--job-name` | *(auto)* | Override inferred job name |
| `--log-dir` | `.slurm_logs` | Initial directory for stdout/stderr logs (also copied to run output dir) |
| `--conda-env` | *(none)* | Conda environment to activate |
| `--venv` | *(auto-detect)* | Path to virtualenv to activate |
| `--module` | *(none)* | Extra `module load` calls (repeatable) |
| `--submit` | `False` | Submit generated script via `sbatch` |
| `--dry-run` | `False` | Print/generate script but do **not** call `sbatch` |
| `--output FILE` | *(stdout)* | Write script to `FILE` instead of stdout |

**Job-name inference** (`infer_job_name()`): experiment config → mode+model+dataset → sanitised fallback.
Sweeps (`-m`) prefix the name with `sweep_` and append `_<index>` per combination.

**Environment activation order** (auto-detect when neither `--conda-env` nor `--venv` given):
1. `.venv/bin/activate` in the project root (relative to `$SLURM_SUBMIT_DIR`)
2. Active conda env at submit time (passed through unchanged)

**SLURM log placement**: SLURM initially writes stdout/stderr to `--log-dir`
(default `.slurm_logs/`).  After `main.py` finishes, the generated script detects the
newly created Hydra output directory under `outputs/` and copies the log there as
`slurm_<JOB_ID>.log`, so each run's output folder is fully self-contained.

---

## 11. Analysis Scripts (`scripts/`)

Standalone Python scripts for post-hoc analysis.  Each script is self-contained
(adds the repo root to `sys.path`) and can be run directly from the server without
going through `main.py`.

### `scripts/plot_checkpoint_evolution_custom.py`

Plot how the generated distribution evolves across selected epoch checkpoints.

**Typical usage:**

```bash
python scripts/plot_checkpoint_evolution_custom.py outputs/generation/mock/noise_10spikes \
    --epochs 9 19 29 39 49 59 69 79 109 209 309 409 509 609 709
```

| CLI flag | Default | Description |
|---|---|---|
| `run_dir` | *(required)* | Path to the run directory |
| `--epochs` | *(required)* | List of epoch numbers to include |
| `--n_generate` | `100000` | Events generated per checkpoint |
| `--bins` | `200` | Histogram bins |

### `scripts/plot_distributions_diff_pdf.py`

Generate a PDF comparing generated vs truth distributions for a run.

```bash
python scripts/plot_distributions_diff_pdf.py outputs/generation/mcpom/mcpom_gen
```

### `scripts/plot_denoise_comparison_pdf.py`

Compare multiple denoising runs side by side.

```bash
python scripts/plot_denoise_comparison_pdf.py \
  outputs/denoise/mcpom/sigma_1.0 \
  outputs/denoise/mcpom/sigma_0.5 \
  outputs/denoise/mcpom/sigma_2.0
```

### `scripts/plot_loss_vs_physics_metrics.py`

Plot training loss vs physics metrics (chi², Wasserstein, etc.) over epochs.

```bash
python scripts/plot_loss_vs_physics_metrics.py outputs/generation/mcpom/mcpom_gen
```

### `scripts/plot_correlation_matrix_heatmap.py`

Visualize the correlation matrix of generated samples.

```bash
python scripts/plot_correlation_matrix_heatmap.py outputs/generation/mcpom/mcpom_gen --sample-size 8000000
```

### `scripts/generate_flow_trajectory_pdf.py`

Visualize ODE flow trajectories from a checkpoint.

```bash
python scripts/generate_flow_trajectory_pdf.py outputs/generation/mock/delta_0/final_model.ckpt
```

### `scripts/plot_t_closeup_from_npz.py`

Generate closeup plot of the Mandelstam t distribution.

```bash
python scripts/plot_t_closeup_from_npz.py outputs/generation/mcpom/mcpom_gen
```

### `scripts/plot_failure_modes.py`

Analyze and visualize failure modes in generated distributions.

```bash
python scripts/plot_failure_modes.py --output figures/failure_modes.pdf
```

### `scripts/eval_best_checkpoint.py`

Evaluate metrics (chi2, W1, chi2_2D, D_corr, R_NN) on the best checkpoint using
pre-generated samples (`generated_samples_best.npz`). Outputs results to
`figures/best_checkpoint_metrics.json`.

### `scripts/prepare_checkpoints_release.sh`

Helper script to organize trained checkpoint files for upload to Zenodo.

---

## 12. Known Patterns & Conventions

### Naming

- **Configs**: `snake_case` for field names; config group names are also `snake_case`.
- **Classes**: `CamelCase`.
- **Private helpers in `main.py`**: prefixed with `_` (e.g. `_write_run_summary`, `_serialize_transform`).
- **Module constants**: `UPPER_CASE` with leading `_` for module-private (e.g. `_MCPOM_SCALES`).

### Paired vs. Unpaired Data

- `BaseDataset.paired=False` → `__getitem__` returns single tensor; model receives Gaussian noise as `x0`.
- `BaseDataset.paired=True` → `__getitem__` returns `(particle_tensor, detector_tensor)`;
  model receives detector data as conditioning `x0`.

### BatchNorm at Inference

Models with `BatchNorm1d` layers must be kept in `train()` mode during generation
(not `eval()`) so that live batch statistics are used. `predict()` handles this
automatically via `has_batchnorm` detection.

### Transform Persistence

The transform fitted on training data is serialised into the checkpoint by
`CheckpointMetadataCallback` → `_serialize_transform` → `transform.serialize()`.
At inference, `_deserialize_transform` restores it and `predict()` applies
`transform.inverse_transform(samples)` to convert back to physical units.

### Dataset Cache

`dataset.save_data(output_dir)` saves `dataset_cache.npz` with keys:
`data`, `original_data`, `detector_data`, `pre_transform_data`, `columns`, `data_dim`.
`predict()` searches parent directories of the checkpoint for this file
to ensure the exact same splits are reused.

### Deduplication

- `BaseDataset._setup_data_from_df(df)` — single implementation of the detector +
  transform pipeline, shared by `MCPom` and all `Synthetic` subclasses.
- `_MCPOM_COLS_LIST` and `_MCPOM_SCALES` module constants are referenced by all four
  `plot_distributions_*_v0` functions instead of defining inline copies.
- `DetectorSmearingConfig` (in `configs/configs.py`) — shared base class carrying
  `sigma: float = MISSING` for both `DetectorMomentumSmearingConfig` and
  `DetectorGeneralSmearingConfig`; eliminates the duplicate field declaration.
- `CheckpointMetadataCallback` / `FirstNEpochsCheckpoint` — defined at module level
  in `main.py` (not inside `train()`), so they can be reused without re-definition.
