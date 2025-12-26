"""Transformer blocks for VAR-Encoder.

Standard Pre-LayerNorm Transformer architecture with:
- Multi-Head Self-Attention (scale-wise causal)
- MLP with GELU activation
- DropPath (stochastic depth)
- Optional RoPE-2D position encoding
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import ScaleCausalAttention, FlashScaleCausalAttention


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


class MLP(nn.Module):
    """MLP block with GELU activation."""

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        out_dim = out_dim or dim

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer block with Pre-LayerNorm."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        use_flash_attn: bool = True,
        use_rope: bool = False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        AttentionClass = FlashScaleCausalAttention if use_flash_attn else ScaleCausalAttention
        self.attn = AttentionClass(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_rope=use_rope,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            dim=dim,
            hidden_dim=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-LN: Norm → Attention → Residual
        x = x + self.drop_path(self.attn(self.norm1(x), coords=coords))
        # Pre-LN: Norm → MLP → Residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Stack of Transformer blocks."""

    def __init__(
        self,
        dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        use_flash_attn: bool = True,
        use_rope: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_rope = use_rope

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                use_flash_attn=use_flash_attn,
                use_rope=use_rope,
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(dim)

    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        return_all_layers: bool = False,
    ) -> torch.Tensor:
        """Forward pass through all transformer blocks.

        Args:
            x: [B, N, D] input tokens
            coords: [340, 2] patch center coordinates (for RoPE)
            return_all_layers: If True, return hidden states from all layers

        Returns:
            If return_all_layers:
                all_hidden: List of [B, N, D] tensors, one per layer + final
            Else:
                hidden: [B, N, D] final hidden states
        """
        if return_all_layers:
            all_hidden = [x]

        for block in self.blocks:
            x = block(x, coords=coords)
            if return_all_layers:
                all_hidden.append(x)

        x = self.norm(x)

        if return_all_layers:
            all_hidden[-1] = x  # Replace last with normalized
            return all_hidden

        return x


if __name__ == "__main__":
    from .patchify import get_all_centers

    # Test transformer block
    block = TransformerBlock(dim=768, num_heads=12)
    x = torch.randn(2, 341, 768)
    out = block(x)
    print(f"Block output shape: {out.shape}")  # [2, 341, 768]

    # Test full encoder
    encoder = TransformerEncoder(dim=768, depth=12, num_heads=12)
    out = encoder(x)
    print(f"Encoder output shape: {out.shape}")  # [2, 341, 768]

    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters: {num_params / 1e6:.2f}M")

    # Test return all layers
    all_hidden = encoder(x, return_all_layers=True)
    print(f"Number of layer outputs: {len(all_hidden)}")  # 13 (input + 12 layers)

    # Test with RoPE
    print("\n--- Testing with RoPE ---")
    encoder_rope = TransformerEncoder(dim=768, depth=12, num_heads=12, use_rope=True)
    coords = get_all_centers()
    out_rope = encoder_rope(x, coords=coords)
    print(f"RoPE encoder output shape: {out_rope.shape}")
