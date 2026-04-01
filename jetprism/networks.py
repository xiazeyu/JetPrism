import logging
import math
import torch
import torch.nn as nn
from typing import List, Union

log = logging.getLogger(__name__)

class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        log.info("BaseNetwork initialized")

class MLP(BaseNetwork):
    """
    Bare MLP backbone. Models wrap this and handle their own conditioning.
    
    Args:
        input_dim: Input dimension (model decides what to concat before passing)
        output_dim: Output dimension
        hidden_dims: Hidden layer dimensions
        activation: "relu", "silu", "gelu", "tanh"
        norm: "layer", "batch", or None
        dropout: Dropout probability
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] | int = 256,
        activation: str = "relu",
        norm: str | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        
        act_fn = {
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }[activation]
        
        layers: list[nn.Module] = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if norm == "layer":
                    layers.append(nn.LayerNorm(dims[i + 1]))
                elif norm == "batch":
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(act_fn())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ResidualNetwork(nn.Module):

    # Creates a residual block
    class ResBlock(nn.Module):
        def __init__(self, 
            input_dim: int, 
            output_dim: int,
            hidden_dim: int = 512,
            activation: str = "relu",
            norm: str | None = None,
            dropout: float = 0.0,
            ):
            super().__init__()
            layers: list[nn.Module] = [] # list of layers
            act_fn = {
                "relu": nn.ReLU,
                "silu": nn.SiLU,
                "gelu": nn.GELU,
                "tanh": nn.Tanh,
            }[activation]
            norm_fn = {
                "layer": nn.LayerNorm,
                "batch": nn.BatchNorm1d,
                None: nn.Identity,
            }[norm]
            
            layers.append(nn.Linear(in_features=input_dim, out_features=hidden_dim))
            layers.append(norm_fn(hidden_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            layers.append(nn.Linear(in_features=hidden_dim, out_features=output_dim))
            layers.append(norm_fn(output_dim))
            layers.append(act_fn())

            if input_dim != output_dim:
                self.skip = nn.Linear(in_features=input_dim, out_features=output_dim)
            else:
                self.skip = nn.Identity()
            
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x) + self.skip(x)
    
    # Actual Model
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: List[int] = [256],
        activation: str = "relu",
        norm: str | None = None,
        dropout: float = 0.0
        ):
        log.info("Instantiating ResidualNetwork")
        super().__init__()

        self.input_dim, self.output_dim, self.hidden_dim = input_dim, output_dim, hidden_dim
        layers: list[nn.Module] = []

        layers.append(nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim[0])) # input layer
        c = self.hidden_dim[0]
        
        for s in self.hidden_dim[1:]:
            layers.append(self.ResBlock(
                input_dim=c, 
                output_dim=s, 
                hidden_dim=c,  # Use current dim as hidden dim in block
                norm=norm, 
                activation=activation,
                dropout=dropout,
            ))
            c = s
        
        layers.append(nn.Linear(in_features=c, out_features=output_dim, bias=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding (used in Diffusion)"""
    def __init__(self, dim: int, max_period: float = 1000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class LearnedEmbedding(nn.Module):
    """Simple learned embedding (often used in Flow Matching)"""
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.net(x)


class FourierEmbedding(nn.Module):
    """Fourier features for continuous time t ∈ [0, 1] (used in Flow Matching).

    Uses fixed sinusoidal bases with geometrically-spaced frequencies covering
    1 to max_freq cycles per unit interval, followed by a learned linear
    projection. This is far superior to LearnedEmbedding for time-conditioned
    velocity fields because the network is given ready-made high-frequency
    features rather than having to rediscover them from a scalar input.

    Args:
        embed_dim: Output feature dimension (must be even).
        max_freq: Highest frequency in cycles per unit interval. Default 64
                  means the fastest basis completes 64 full oscillations over
                  t ∈ [0, 1], resolving velocity changes at scale ~1/64.
    """
    def __init__(self, embed_dim: int, max_freq: float = 64.0):
        super().__init__()
        half = embed_dim // 2
        # Geometrically spaced: 1, ..., max_freq cycles/unit * 2π rad/cycle
        freqs = torch.exp(
            torch.linspace(0.0, math.log(max_freq), half)
        ) * 2.0 * math.pi
        self.register_buffer('freqs', freqs)  # fixed, not trained
        self.proj = nn.Linear(embed_dim, embed_dim)  # learned mixing
        self.dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)                       # [B, 1]
        args = t * self.freqs[None, :]                # [B, half]
        fourier = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, embed_dim]
        return self.proj(fourier)                     # [B, embed_dim]

class DiffusionMLP(BaseNetwork):
    """
    MLP for Diffusion models.
    Uses sinusoidal time embedding.
    """
    def __init__(
        self,
        data_dim: int,
        time_embed_dim: int = 64,
        hidden_dims: Union[List[int], int] = 256,
        **mlp_kwargs,
    ):
        super().__init__()
        self.time_embed = SinusoidalEmbedding(time_embed_dim)
        self.time_proj = nn.Linear(time_embed_dim, time_embed_dim)
        self.net = MLP(
            input_dim=data_dim + time_embed_dim,
            output_dim=data_dim,
            hidden_dims=hidden_dims,
            **mlp_kwargs,
        )
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy data [batch, data_dim]
            t: Diffusion timestep [batch]
        Returns:
            Predicted noise or score [batch, data_dim]
        """
        t_embed = self.time_proj(self.time_embed(t))
        return self.net(torch.cat([x, t_embed], dim=-1))

class FlowMatchingMLP(BaseNetwork):
    """
    MLP for Flow Matching / Conditional Flow Matching.
    Fourier time embedding for accurate conditioning on t ∈ [0, 1].
    """
    def __init__(
        self,
        data_dim: int,
        time_embed_dim: int = 64,
        hidden_dims: Union[List[int], int] = 256,
        use_ot_cost: bool = False,  # FM-specific: include OT cost as input
        time_embedding: str = "fourier",  # "fourier" or "learned" (legacy)
        **mlp_kwargs,
    ):
        super().__init__()
        self.use_ot_cost = use_ot_cost
        if time_embedding == "learned":
            self.time_embed = LearnedEmbedding(1, time_embed_dim)
        else:
            self.time_embed = FourierEmbedding(time_embed_dim)
        
        extra_dim = 1 if use_ot_cost else 0
        self.net = MLP(
            input_dim=data_dim + time_embed_dim + extra_dim,
            output_dim=data_dim,
            hidden_dims=hidden_dims,
            **mlp_kwargs,
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor,
        ot_cost: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Interpolated point x_t [batch, data_dim]
            t: Time in [0, 1] [batch]
            ot_cost: Optional OT cost for CFM [batch, 1]
        Returns:
            Predicted velocity [batch, data_dim]
        """
        t_embed = self.time_embed(t)
        inputs = [x, t_embed]
        if self.use_ot_cost and ot_cost is not None:
            inputs.append(ot_cost)
        return self.net(torch.cat(inputs, dim=-1))


class FlowMatchingResNet(BaseNetwork):
    """
    ResNet for Flow Matching / Conditional Flow Matching.
    Fourier time embedding for accurate conditioning on t ∈ [0, 1].
    Uses ResidualNetwork instead of MLP for better gradient flow.
    """
    def __init__(
        self,
        data_dim: int,
        time_embed_dim: int = 64,
        hidden_dims: Union[List[int], int] = 256,
        use_ot_cost: bool = False,  # FM-specific: include OT cost as input
        time_embedding: str = "fourier",  # "fourier" or "learned" (legacy)
        **resnet_kwargs,
    ):
        super().__init__()
        self.use_ot_cost = use_ot_cost
        if time_embedding == "learned":
            self.time_embed = LearnedEmbedding(1, time_embed_dim)
        else:
            self.time_embed = FourierEmbedding(time_embed_dim)
        
        extra_dim = 1 if use_ot_cost else 0
        
        # Convert hidden_dims to list format expected by ResidualNetwork
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims, hidden_dims]
        elif isinstance(hidden_dims, list) and len(hidden_dims) == 1:
            hidden_dims = hidden_dims + hidden_dims  # Need at least 2 for ResBlocks
        
        self.net = ResidualNetwork(
            input_dim=data_dim + time_embed_dim + extra_dim,
            output_dim=data_dim,
            hidden_dim=hidden_dims,
            **resnet_kwargs,
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor,
        ot_cost: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Interpolated point x_t [batch, data_dim]
            t: Time in [0, 1] [batch]
            ot_cost: Optional OT cost for CFM [batch, 1]
        Returns:
            Predicted velocity [batch, data_dim]
        """
        t_embed = self.time_embed(t)
        inputs = [x, t_embed]
        if self.use_ot_cost and ot_cost is not None:
            inputs.append(ot_cost)
        return self.net(torch.cat(inputs, dim=-1))


class ConditionalMLP(BaseNetwork):
    """
    MLP with generic conditioning (class labels, context, etc.)
    Works with both Diffusion and Flow Matching.
    """
    
    time_embed: nn.Module  # Type hint for mixed types (Sequential or LearnedEmbedding)
    
    def __init__(
        self,
        data_dim: int,
        cond_dim: int,
        time_embed_dim: int = 64,
        cond_embed_dim: int = 64,
        hidden_dims: Union[List[int], int] = 256,
        time_embedding: str = "sinusoidal",  # "sinusoidal" or "learned"
        **mlp_kwargs,
    ):
        super().__init__()
        
        if time_embedding == "sinusoidal":
            self.time_embed = nn.Sequential(
                SinusoidalEmbedding(time_embed_dim),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
        else:
            self.time_embed = LearnedEmbedding(1, time_embed_dim)
            
        self.cond_embed = nn.Linear(cond_dim, cond_embed_dim)
        
        self.net = MLP(
            input_dim=data_dim + time_embed_dim + cond_embed_dim,
            output_dim=data_dim,
            hidden_dims=hidden_dims,
            **mlp_kwargs,
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        cond: torch.Tensor,
    ) -> torch.Tensor:
        t_embed = self.time_embed(t)
        c_embed = self.cond_embed(cond)
        return self.net(torch.cat([x, t_embed, c_embed], dim=-1))


class ConditionalFlowMatchingMLP(BaseNetwork):
    """
    MLP for Conditional Flow Matching with explicit conditioning input.
    Used for denoising/unfolding tasks where detector data conditions the generation.
    
    Input: (x_t, t, condition) -> predicted velocity
    Where x_t is interpolated noise-to-target, and condition is the detector data.
    """
    def __init__(
        self,
        data_dim: int,
        cond_dim: int,
        time_embed_dim: int = 64,
        cond_embed_dim: int | None = None,
        hidden_dims: Union[List[int], int] = 256,
        time_embedding: str = "fourier",
        **mlp_kwargs,
    ):
        super().__init__()
        
        # Use same embedding dim as data if not specified
        if cond_embed_dim is None:
            cond_embed_dim = data_dim
        
        if time_embedding == "learned":
            self.time_embed = LearnedEmbedding(1, time_embed_dim)
        else:
            self.time_embed = FourierEmbedding(time_embed_dim)
        
        # Embed conditioning (detector data) to a learned representation
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
        )
        
        self.net = MLP(
            input_dim=data_dim + time_embed_dim + cond_embed_dim,
            output_dim=data_dim,
            hidden_dims=hidden_dims,
            **mlp_kwargs,
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Interpolated point x_t [batch, data_dim]
            t: Time in [0, 1] [batch]
            cond: Conditioning data (detector-level) [batch, cond_dim]
        Returns:
            Predicted velocity [batch, data_dim]
        """
        t_embed = self.time_embed(t)
        c_embed = self.cond_embed(cond)
        return self.net(torch.cat([x, t_embed, c_embed], dim=-1))


class ConditionalFlowMatchingResNet(BaseNetwork):
    """
    ResNet for Conditional Flow Matching with explicit conditioning input.
    Used for denoising/unfolding tasks where detector data conditions the generation.
    
    Input: (x_t, t, condition) -> predicted velocity
    Where x_t is interpolated noise-to-target, and condition is the detector data.
    """
    def __init__(
        self,
        data_dim: int,
        cond_dim: int,
        time_embed_dim: int = 64,
        cond_embed_dim: int | None = None,
        hidden_dims: Union[List[int], int] = 256,
        time_embedding: str = "fourier",
        **resnet_kwargs,
    ):
        super().__init__()
        
        # Use same embedding dim as data if not specified
        if cond_embed_dim is None:
            cond_embed_dim = data_dim
        
        if time_embedding == "learned":
            self.time_embed = LearnedEmbedding(1, time_embed_dim)
        else:
            self.time_embed = FourierEmbedding(time_embed_dim)
        
        # Embed conditioning (detector data) to a learned representation
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
        )
        
        # Convert hidden_dims to list format expected by ResidualNetwork
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims, hidden_dims]
        elif isinstance(hidden_dims, list) and len(hidden_dims) == 1:
            hidden_dims = hidden_dims + hidden_dims
        
        self.net = ResidualNetwork(
            input_dim=data_dim + time_embed_dim + cond_embed_dim,
            output_dim=data_dim,
            hidden_dim=hidden_dims,
            **resnet_kwargs,
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Interpolated point x_t [batch, data_dim]
            t: Time in [0, 1] [batch]
            cond: Conditioning data (detector-level) [batch, cond_dim]
        Returns:
            Predicted velocity [batch, data_dim]
        """
        t_embed = self.time_embed(t)
        c_embed = self.cond_embed(cond)
        return self.net(torch.cat([x, t_embed, c_embed], dim=-1))