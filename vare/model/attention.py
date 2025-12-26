"""Scale-wise causal attention mask for VAR-Encoder.

Attention rules:
- CLS token: Sees ALL tokens (global aggregation)
- Scale k tokens: Can see CLS + all previous scales + current scale (bidirectional within scale)

Mask structure (341 x 341):
         CLS │ S1(4) │ S2(16) │ S3(64) │ S4(256)
        ─────┼───────┼────────┼────────┼─────────
   CLS │  ✓  │   ✓   │   ✓    │   ✓    │   ✓    ← CLS sees all!
   S1  │  ✓  │   ✓   │   ✗    │   ✗    │   ✗
   S2  │  ✓  │   ✓   │   ✓    │   ✗    │   ✗
   S3  │  ✓  │   ✓   │   ✓    │   ✓    │   ✗
   S4  │  ✓  │   ✓   │   ✓    │   ✓    │   ✓

Supports:
- Standard attention with additive position embeddings
- RoPE-2D (Rotary Position Embedding) applied to Q, K
"""

import torch
import torch.nn as nn
from typing import List, Optional
from functools import lru_cache

from .patchify import SCALE_CONFIGS, NUM_SCALES, TOTAL_PATCH_TOKENS, RoPE2D


# Token counts: [CLS, S1, S2, S3, S4]
TOKENS_PER_SCALE = [1] + [cfg[2] for cfg in SCALE_CONFIGS]  # [1, 4, 16, 64, 256]
TOTAL_TOKENS = sum(TOKENS_PER_SCALE)  # 341


@lru_cache(maxsize=1)
def build_scale_causal_mask(device: str = 'cpu') -> torch.Tensor:
    """Build scale-wise causal attention mask.

    Returns:
        mask: [341, 341] attention mask
            - 0 where attention is allowed
            - -inf where attention is blocked

    Special: CLS (index 0) can attend to ALL tokens (global aggregation).
    """
    total = TOTAL_TOKENS
    mask = torch.zeros(total, total)

    # Compute cumulative indices
    cumsum = [0]
    for n in TOKENS_PER_SCALE:
        cumsum.append(cumsum[-1] + n)
    # cumsum = [0, 1, 5, 21, 85, 341]

    # For each query scale i, mask out future scales j > i
    # Skip i=0 (CLS) since CLS can see everything
    for i in range(1, len(TOKENS_PER_SCALE)):  # Start from 1, not 0
        start_i, end_i = cumsum[i], cumsum[i + 1]
        for j in range(len(TOKENS_PER_SCALE)):
            start_j, end_j = cumsum[j], cumsum[j + 1]
            if j > i:
                # Query from scale i cannot attend to keys from future scale j
                mask[start_i:end_i, start_j:end_j] = float('-inf')

    # CLS (row 0) can attend to everything - already 0, no change needed

    return mask.to(device)


def get_scale_boundaries() -> List[int]:
    """Get token index boundaries for each scale.

    Returns:
        [0, 1, 5, 21, 85, 341] - boundaries including CLS
    """
    boundaries = [0]
    for n in TOKENS_PER_SCALE:
        boundaries.append(boundaries[-1] + n)
    return boundaries


class ScaleCausalAttention(nn.Module):
    """Multi-head self-attention with scale-wise causal masking.

    Supports optional RoPE-2D for position encoding.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rope: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # RoPE for each head dimension
        if use_rope:
            self.rope = RoPE2D(self.head_dim)
        else:
            self.rope = None

        # Cache for attention mask
        self._attn_mask: Optional[torch.Tensor] = None

    def _get_mask(self, device: torch.device) -> torch.Tensor:
        """Get or create attention mask for the given device."""
        if self._attn_mask is None or self._attn_mask.device != device:
            self._attn_mask = build_scale_causal_mask(str(device))
        return self._attn_mask

    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with scale-wise causal attention.

        Args:
            x: [B, N, D] where N = 341 (CLS + 340 patches)
            coords: [340, 2] patch center coordinates (required if use_rope=True)

        Returns:
            out: [B, N, D]
        """
        B, N, D = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv.unbind(0)

        # Apply RoPE if enabled
        if self.use_rope and coords is not None:
            # q, k: [B, heads, N, head_dim]
            # Need to reshape for RoPE which expects [B, N, D]
            B, H, N, HD = q.shape

            # Reshape: [B, H, N, HD] → [B*H, N, HD]
            q = q.reshape(B * H, N, HD)
            k = k.reshape(B * H, N, HD)

            # Apply RoPE (skip first token = CLS)
            q = self.rope(q, coords, skip_first=True)
            k = self.rope(k, coords, skip_first=True)

            # Reshape back: [B*H, N, HD] → [B, H, N, HD]
            q = q.reshape(B, H, N, HD)
            k = k.reshape(B, H, N, HD)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]

        # Apply scale-wise causal mask
        mask = self._get_mask(x.device)
        attn = attn + mask.unsqueeze(0).unsqueeze(0)

        # Softmax and dropout
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)

        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class FlashScaleCausalAttention(nn.Module):
    """Scale-wise causal attention using PyTorch's scaled_dot_product_attention.

    More memory-efficient for longer sequences.
    Supports optional RoPE-2D.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rope: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if use_rope:
            self.rope = RoPE2D(self.head_dim)
        else:
            self.rope = None

        self._attn_mask: Optional[torch.Tensor] = None

    def _get_mask(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get attention mask as boolean or float depending on SDPA requirements."""
        if self._attn_mask is None or self._attn_mask.device != device:
            mask = build_scale_causal_mask(str(device))
            # Convert to boolean mask (True = masked/blocked)
            self._attn_mask = mask == float('-inf')
        return self._attn_mask

    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Apply RoPE if enabled
        if self.use_rope and coords is not None:
            B, H, N, HD = q.shape
            q = q.reshape(B * H, N, HD)
            k = k.reshape(B * H, N, HD)
            q = self.rope(q, coords, skip_first=True)
            k = self.rope(k, coords, skip_first=True)
            q = q.reshape(B, H, N, HD)
            k = k.reshape(B, H, N, HD)

        # Get boolean mask
        mask = self._get_mask(x.device, x.dtype)

        # Use scaled_dot_product_attention
        dropout_p = self.attn_drop if self.training else 0.0
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=~mask,  # SDPA expects True = attend, False = mask
            dropout_p=dropout_p,
        )

        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


if __name__ == "__main__":
    from .patchify import get_all_centers

    # Test mask construction
    mask = build_scale_causal_mask()
    print(f"Mask shape: {mask.shape}")  # [341, 341]

    # Visualize mask structure
    boundaries = get_scale_boundaries()
    print(f"Scale boundaries: {boundaries}")

    # Check that CLS only sees itself
    print(f"\nCLS row (should see only itself): {mask[0, :5]}")

    # Check that S1 sees CLS + S1
    print(f"S1 first token row: {mask[1, :10]}")

    # Check that S4 sees everything
    print(f"S4 first token row (first 10): {mask[85, :10]}")

    # Test attention without RoPE
    print("\n--- Testing Standard Attention ---")
    attn = ScaleCausalAttention(dim=768, num_heads=12, use_rope=False)
    x = torch.randn(2, 341, 768)
    out = attn(x)
    print(f"Attention output shape: {out.shape}")  # [2, 341, 768]

    # Test attention with RoPE
    print("\n--- Testing Attention with RoPE ---")
    attn_rope = ScaleCausalAttention(dim=768, num_heads=12, use_rope=True)
    coords = get_all_centers()
    out_rope = attn_rope(x, coords=coords)
    print(f"RoPE attention output shape: {out_rope.shape}")  # [2, 341, 768]
