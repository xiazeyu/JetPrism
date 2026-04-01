from dataclasses import dataclass, field
from typing import Any
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
import torch
from jetprism import schemas

import logging

log = logging.getLogger(__name__)
cs = ConfigStore.instance()


def register_config(group: str, name: str = "default", node: Any = None):
    """
    Decorator/Helper to register a config node in the ConfigStore.
    Can be used as @register_config(group="...", name="...") or called directly.
    """
    def _register(cls):
        cs.store(group=group, name=name, node=cls)
        return cls

    if node is not None:
        # If called as a function (not decorator)
        cs.store(group=group, name=name, node=node)
        return node

    return _register


# ─── Path ─────────────────────────────────────────────────────────────────────

@register_config(group="path")
@dataclass
class PathConfig:
    cwd: str = "${hydra:runtime.cwd}"
    output_dir: str | None = "${hydra:runtime.output_dir}"
    data_dir: str = "${hydra:runtime.cwd}/data"


@register_config(group="trainer")
@dataclass
class TrainerConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 800
    log_interval: int = 10  # Save a checkpoint and log metrics every N epochs
    val_interval: int = 10  # Validate every N epochs (1 = every epoch)
    num_sanity_val_steps: int = 1
    save_first_n_checkpoints: int = 0  # Also checkpoint at epoch-end for the first N epochs (0 = disabled)
    plot_checkpoint_evolution: bool = False  # Run checkpoint-evolution plots after training
    evolution_samples: int = 50000  # Samples per checkpoint for post-training evolution plot
    evolution_every_n: int = 1  # Plot every N-th checkpoint in evolution


# ─── Datasets ─────────────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    _target_: str = MISSING
    batch_size: int = 20000
    shuffle: bool = True
    num_workers: int = 4
    random_seed: int | None = 42
    split_ratios: tuple[float, float, float] = field(
        default_factory=lambda: (1.0, 0.0, 0.0))
    detector: Any = "${detector}"
    transform: Any = "${transform}"
    data_dir: str = "${path.data_dir}"
    paired: bool = False


@register_config(group="dataset", name="mcpom")
@dataclass
class MCPomConfig(DatasetConfig):
    _target_: str = "jetprism.datasets.MCPom"
    file_name: str = 'mc_pom_v2.parquet'
    sample_num: int | None = None
    paired: bool = False


@dataclass
class SyntheticConfig(DatasetConfig):
    sample_num: int = 8_000_000
    dim: int = 1
    columns: list[str] | None = None


@register_config(group="dataset", name="gaussian")
@dataclass
class GaussianConfig(SyntheticConfig):
    _target_: str = "jetprism.datasets.Gaussian"
    mean: float = 0.0
    std: float = 1.0


@register_config(group="dataset", name="highcut")
@dataclass
class HighCut(SyntheticConfig):
    _target_: str = "jetprism.datasets.HighCut"
    mean: float = 0.0
    std: float = 1.0
    threshold: float = 0.7
    buffer_multiplier: float = 2.0


@register_config(group="dataset", name="multipeak")
@dataclass
class MultiPeak(SyntheticConfig):
    _target_: str = "jetprism.datasets.MultiPeak"
    peaks: list = field(default_factory=lambda: [
        # [mean, std, weight]
        [-1.0, 0.5, 0.3], [1.0, 0.5, 0.7]])


@register_config(group="dataset", name="highfreq")
@dataclass
class HighFrequency(SyntheticConfig):
    _target_: str = "jetprism.datasets.HighFrequency"
    base_mean: float = 0.0
    base_std: float = 1.0

    noise_prob: float = 0.1  # 10% of data comes from the "noise" peaks
    num_noise_peaks: int = 20
    noise_std: float = 0.05  # Very sharp peaks
    # Where the noise peaks are scattered
    noise_range: list = field(default_factory=lambda: [-3.0, 3.0])


@register_config(group="dataset", name="delta")
@dataclass
class DeltaFunction(SyntheticConfig):
    _target_: str = "jetprism.datasets.DeltaFunction"
    # Delta function centered at this value (default 0)
    center: float = 0.0


@register_config(group="dataset", name="uniform")
@dataclass
class UniformConfig(SyntheticConfig):
    _target_: str = "jetprism.datasets.Uniform"
    low: float = -1.0
    high: float = 1.0


@register_config(group="dataset", name="exponential")
@dataclass
class ExponentialConfig(SyntheticConfig):
    _target_: str = "jetprism.datasets.Exponential"
    scale: float = 1.0
    loc: float = 0.0


# ─── Detectors ───────────────────────────────────────────────────────────────

@dataclass
class DetectorConfig:
    _target_: str = MISSING


@register_config(group="detector", name="compose")
@dataclass
class DetectorComposeConfig(DetectorConfig):
    _target_: str = "jetprism.detectors.Compose"
    _recursive_: bool = False
    detectors: list = MISSING


@register_config(group="detector", name="identity")
@dataclass
class DetectorIdentityConfig(DetectorConfig):
    _target_: str = "jetprism.detectors.Identity"


@register_config(group="detector", name="cos_theta_cut")
@dataclass
class DetectorCosThetaCutConfig(DetectorConfig):
    _target_: str = "jetprism.detectors.CosThetaCut"
    threshold: float = MISSING


@register_config(group="detector", name="value_cut")
@dataclass
class DetectorValueCutConfig(DetectorConfig):
    _target_: str = "jetprism.detectors.ValueCut"
    column: str = MISSING
    min_value: float | None = MISSING
    max_value: float | None = MISSING


@dataclass
class DetectorSmearingConfig(DetectorConfig):
    """Shared base for smearing detectors; subclasses set _target_ and inherit sigma."""
    sigma: float = MISSING
    random_seed: int | None = 42


@register_config(group="detector", name="momentum_smearing")
@dataclass
class DetectorMomentumSmearingConfig(DetectorSmearingConfig):
    _target_: str = "jetprism.detectors.MomentumSmearing"


@register_config(group="detector", name="general_smearing")
@dataclass
class DetectorGeneralSmearingConfig(DetectorSmearingConfig):
    _target_: str = "jetprism.detectors.GeneralSmearing"


@register_config(group="detector", name="uniform_phi")
@dataclass
class DetectorUniformPhiConfig(DetectorConfig):
    _target_: str = "jetprism.detectors.UniformPhi"
    num_bins: int = 50
    random_state: int | None = 42


# ─── Transforms ──────────────────────────────────────────────────────────────

@dataclass
class TransformConfig:
    _target_: str = MISSING


@register_config(group="transform", name="compose")
@dataclass
class TransformComposeConfig(TransformConfig):
    _target_: str = "jetprism.transforms.Compose"
    _recursive_: bool = False
    transforms: list = MISSING


@register_config(group="transform", name="identity")
@dataclass
class TransformIdentityConfig(TransformConfig):
    _target_: str = "jetprism.transforms.Identity"


@register_config(group="transform", name="standard_scaler")
@dataclass
class TransformStandardScalerConfig(TransformConfig):
    _target_: str = "jetprism.transforms.StandardScaler"
    mean: list[float] | None = None
    std: list[float] | None = None
    scale: float = 1.0


@register_config(group="transform", name="log_transformer")
@dataclass
class TransformLogTransformerConfig(TransformConfig):
    _target_: str = "jetprism.transforms.LogTransformer"
    columns: list[int] | None = None
    offset: float = 1.0


@register_config(group="transform", name="four_particle_representation")
@dataclass
class TransformFourParticleRepresentationConfig(TransformConfig):
    _target_: str = "jetprism.transforms.FourParticleRepresentation"


@register_config(group="transform", name="dlpp_representation")
@dataclass
class TransformDLPPRepresentationConfig(TransformConfig):
    _target_: str = "jetprism.transforms.DLPPRepresentation"

# ─── Models ──────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    _target_: str = MISSING
    data_dim: int | None = None  # Auto-inferred from dataset if None
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: str = "plateau"  # "plateau", "cosine", "none"
    scheduler_patience: int = 50  # Reduced for faster response with val_interval=1
    scheduler_factor: float = 0.5
    grad_clip_val: float | None = None
    metric_sample_size: int = 1000000  # Events used for chi2/KS/Wasserstein at validation


@register_config(group="model", name="cfm")
@dataclass
class CFMConfig(ModelConfig):
    _target_: str = "jetprism.models.CFM"
    hidden_dims: list[int] = field(default_factory=lambda: [512, 512, 512, 512, 512, 512])
    time_embed_dim: int = 64
    sigma: float = 0.0
    solver: str = "dopri5"
    solver_atol: float = 1e-5
    solver_rtol: float = 1e-5
    solver_steps: int = 100  # Fixed steps for euler/midpoint/rk4
    # Network options
    network_type: str = "resnet"  # "mlp" or "resnet"
    activation: str = "silu"  # "relu", "silu", "gelu", "tanh"
    norm: str | None = None  # "layer", "batch", or None
    dropout: float = 0.0
    # Conditional mode for denoising/unfolding
    conditional: bool = False
    cond_dim: int | None = None  # Defaults to data_dim if not specified
    cond_embed_dim: int | None = None  # Embedding dimension for conditioning


@register_config(group="model", name="ddpm")
@dataclass
class DDPMConfig(ModelConfig):
    _target_: str = "jetprism.models.DDPM"
    hidden_dims: list[int] = field(default_factory=lambda: [512, 512, 512, 512, 512, 512])
    time_embed_dim: int = 64
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "linear"  # "linear", "cosine"
    prediction_type: str = "epsilon"  # "epsilon", "x0", "v"
    # Network options
    activation: str = "silu"  # "relu", "silu", "gelu", "tanh"
    norm: str | None = None  # "layer", "batch", or None
    dropout: float = 0.0


# ─── Root config ─────────────────────────────────────────────────────────────

@dataclass
class Config:
    path: PathConfig = MISSING
    trainer: TrainerConfig = MISSING
    dataset: DatasetConfig = MISSING
    detector: DetectorConfig | None = None
    model: ModelConfig = MISSING
    transform: TransformConfig | None = None
    experiment: Any | None = None

    mode: schemas.Mode = MISSING

    # ── TRAIN ──────────────────────────────────────────────────────────────────
    predict: bool = False  # Run predict after training finishes (mode=TRAIN only)

    # ── PREDICT ────────────────────────────────────────────────────────────────
    checkpoint_path: str | None = None  # Path to .ckpt file
    n_generate: int = 8000000             # Samples to generate in PREDICT mode
    generation_batch_size: int = 20000    # Per-batch generation size (controls VRAM)
    fit_gaussian: bool = False            # Fit Gaussians to generated samples and compare with truth
    save_samples: str = "all"            # Save generated_samples npz: "none", "best", "all"
    predict_best_only: bool = False        # If True, predict only the best checkpoint (skip intermediates and last)

    # ── BATCH_PREDICT / CHECKPOINT_EVOLUTION (batch mode) ─────────────────────
    runs_dir: str | None = None        # Path to a multirun directory

    # ── TEST_FLOW ──────────────────────────────────────────────────────────────
    flow_n_generate: int = 20000         # Trajectory samples to visualise
    flow_num_steps: int = 100             # ODE integration steps
    flow_dims: list = field(default_factory=lambda: [0, 1])  # Dimension pair for 2-D plots
    flow_plot_type: str = "all"           # 'scatter', 'density', 'marginal', or 'all'

    # ── CHECKPOINT_EVOLUTION ──────────────────────────────────────────────────
    evolution_samples: int = 50000        # Samples per checkpoint
    evolution_bins: int = 200             # Histogram bins
    evolution_xmin: float | None = None  # x-axis lower bound (None = auto)
    evolution_xmax: float | None = None  # x-axis upper bound (None = auto)
    evolution_skip_last: bool = False     # Skip last.ckpt in evolution
    evolution_every_n: int = 1            # Plot every N-th checkpoint

    # ── NN MEMORIZATION (PREDICT mode) ────────────────────────────────────────
    compute_nn: bool = True               # Compute NN memorization metric after generation
    nn_sample_size: int = 80_000          # Events sampled for each NN distance computation
    nn_query_batch_size: int = 1_000      # Query batch size (affects peak VRAM)
    nn_ref_batch_size: int = 50_000       # Reference batch size (affects peak VRAM)


cs.store(name="base_config", node=Config)
