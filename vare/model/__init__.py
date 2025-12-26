"""VAR-Encoder model components."""

from .vare import (
    VAREncoder,
    VAREncoderConfig,
    PredictionHead,
    vare_base,
    vare_large,
    vare_huge,
)
from .patchify import (
    MultiScalePatchEmbed,
    MultiScaleTokenizer,  # Alias for backward compatibility
    ContinuousPositionEncoder,
    RoPE2D,
    patchify_single_scale,
    get_patch_centers,
    get_scale_boundaries,
    get_parent_child_mapping,
    get_all_centers,
    SCALE_CONFIGS,
    NUM_SCALES,
    TOTAL_PATCH_TOKENS,
)
from .attention import (
    ScaleCausalAttention,
    FlashScaleCausalAttention,
    build_scale_causal_mask,
    TOKENS_PER_SCALE,
    TOTAL_TOKENS,
)
from .transformer import (
    TransformerBlock,
    TransformerEncoder,
    MLP,
    DropPath,
)

__all__ = [
    # Main model
    "VAREncoder",
    "VAREncoderConfig",
    "PredictionHead",
    "vare_base",
    "vare_large",
    "vare_huge",
    # Patchify
    "MultiScalePatchEmbed",
    "MultiScaleTokenizer",
    "ContinuousPositionEncoder",
    "RoPE2D",
    "patchify_single_scale",
    "get_patch_centers",
    "get_scale_boundaries",
    "get_parent_child_mapping",
    "get_all_centers",
    "SCALE_CONFIGS",
    "NUM_SCALES",
    "TOTAL_PATCH_TOKENS",
    # Attention
    "ScaleCausalAttention",
    "FlashScaleCausalAttention",
    "build_scale_causal_mask",
    "TOKENS_PER_SCALE",
    "TOTAL_TOKENS",
    # Transformer
    "TransformerBlock",
    "TransformerEncoder",
    "MLP",
    "DropPath",
]
