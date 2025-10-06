from typing import Dict


class PentachoronFlowConfig:
    """Configuration for PentachoronFlow network."""

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
        image_size: int = 32,
        patch_size: int = 4,
        num_origins: int = 49,
        origin_dim: int = 5,
        embed_dim: int = 768,
        init_strategy: str = 'diffusion',
        collection_strategy: str = 'geometric_distance',
        adaptive_radius: bool = True,
        base_radius: float = 1.0,
        k_nearest: int = 16,
        flow_steps: int = 4,
        diffusion_steps: int = 1000,
        hidden_scale: int = 4,
        max_grad_norm: float = 1.0,
        validation_weights: Dict[str, float] = None,
        sample_size: float = 0.15,
        use_attention: bool = True,
        attention_heads: int = 8,
        dropout_rate: float = 0.1,
        temperature: float = 1.0,
        quality_threshold: float = 0.5,
        max_init_attempts: int = 10,
        use_trajectory_attention: bool = False,
        trajectory_attention_heads: int = 4,
    ):
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_origins = num_origins
        self.origin_dim = origin_dim
        self.embed_dim = embed_dim
        self.init_strategy = init_strategy
        self.collection_strategy = collection_strategy
        self.adaptive_radius = adaptive_radius
        self.base_radius = base_radius
        self.k_nearest = k_nearest
        self.flow_steps = flow_steps
        self.hidden_scale = hidden_scale
        self.max_grad_norm = max_grad_norm
        self.sample_size = sample_size
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.temperature = temperature
        self.quality_threshold = quality_threshold
        self.max_init_attempts = max_init_attempts
        self.use_trajectory_attention = use_trajectory_attention
        self.trajectory_attention_heads = trajectory_attention_heads
        self.diffusion_steps = diffusion_steps



        if validation_weights is None:
            self.validation_weights = {'rose': 0.5, 'quality': 0.3, 'volume': 0.2}
        else:
            self.validation_weights = validation_weights
