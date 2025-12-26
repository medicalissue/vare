"""Multi-scale patchification for VAR-Encoder.

Converts an image into multi-scale patch embeddings:
1. tokens: CLS + patch embeddings with position encoding (Transformer input)
2. targets: Raw patch embeddings without position (for prediction loss)

Token structure:
- Scale 1: 32×32 → 2×2 patches → 4 tokens
- Scale 2: 64×64 → 4×4 patches → 16 tokens
- Scale 3: 128×128 → 8×8 patches → 64 tokens
- Scale 4: 256×256 → 16×16 patches → 256 tokens
Total: 340 patch tokens (+ 1 CLS = 341)

Key Design:
- Unified [0,1]² coordinate system for all scales
- CLS token has NO position embedding (global, unbiased)
- Position encoding: MLP or RoPE-2D
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# Scale configurations: (image_size, grid_size, num_tokens)
SCALE_CONFIGS = [
    (32, 2, 4),      # Scale 1: 32×32 → 2×2 = 4 tokens
    (64, 4, 16),     # Scale 2: 64×64 → 4×4 = 16 tokens
    (128, 8, 64),    # Scale 3: 128×128 → 8×8 = 64 tokens
    (256, 16, 256),  # Scale 4: 256×256 → 16×16 = 256 tokens
]

NUM_SCALES = len(SCALE_CONFIGS)
TOTAL_PATCH_TOKENS = sum(cfg[2] for cfg in SCALE_CONFIGS)  # 340


def get_patch_centers(grid_size: int) -> torch.Tensor:
    """Compute center coordinates for each patch in [0,1]² space.

    Args:
        grid_size: Number of patches per side (e.g., 2, 4, 8, 16)

    Returns:
        centers: [grid_size², 2] tensor of (x, y) coordinates
    """
    step = 1.0 / grid_size
    offset = step / 2

    centers = []
    for row in range(grid_size):
        for col in range(grid_size):
            x = offset + col * step
            y = offset + row * step
            centers.append([x, y])

    return torch.tensor(centers, dtype=torch.float32)


def patchify_single_scale(
    images: torch.Tensor,
    patch_size: int = 16,
) -> torch.Tensor:
    """Convert images to patch sequence for a single scale."""
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0

    num_h = H // patch_size
    num_w = W // patch_size

    patches = images.reshape(B, C, num_h, patch_size, num_w, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5)
    patches = patches.reshape(B, num_h * num_w, -1)

    return patches


def get_scale_boundaries() -> List[int]:
    """Get token index boundaries for each scale.

    Returns:
        [0, 1, 5, 21, 85, 341] - cumulative indices including CLS
        - CLS: [0:1]
        - S1: [1:5]
        - S2: [5:21]
        - S3: [21:85]
        - S4: [85:341]
    """
    boundaries = [0, 1]
    for _, _, num_tokens in SCALE_CONFIGS:
        boundaries.append(boundaries[-1] + num_tokens)
    return boundaries


def get_parent_child_mapping() -> List[Tuple[int, int, int, int]]:
    """Get parent-child index mappings for 1→4 prediction.

    Returns:
        List of (parent_start, parent_end, child_start, child_end) tuples.
        Indices are in the token sequence (including CLS at 0).

        Mapping (CLS→S1 removed since CLS sees all scales):
        - S1 [1:5] → S2 [5:21]
        - S2 [5:21] → S3 [21:85]
        - S3 [21:85] → S4 [85:341]
    """
    boundaries = get_scale_boundaries()
    mappings = []
    # Start from i=1 (skip CLS→S1 since CLS now sees everything)
    for i in range(1, len(boundaries) - 2):
        p_start = boundaries[i]
        p_end = boundaries[i + 1]
        c_start = boundaries[i + 1]
        c_end = boundaries[i + 2]
        mappings.append((p_start, p_end, c_start, c_end))
    return mappings


def get_all_centers() -> torch.Tensor:
    """Get all patch centers concatenated.

    Returns:
        [340, 2] tensor of all patch center coordinates
    """
    all_centers = []
    for _, grid_size, _ in SCALE_CONFIGS:
        centers = get_patch_centers(grid_size)
        all_centers.append(centers)
    return torch.cat(all_centers, dim=0)


class ContinuousPositionEncoder(nn.Module):
    """Encode 2D coordinates into position embeddings via MLP."""

    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or dim // 2

        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.mlp(coords)


class RoPE2D(nn.Module):
    """2D Rotary Position Embedding."""

    def __init__(self, dim: int, max_freq: float = 10.0):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq

        assert dim % 4 == 0, f"dim must be divisible by 4 for RoPE2D, got {dim}"
        self.half_dim = dim // 4

        freqs = torch.exp(
            torch.arange(0, self.half_dim, dtype=torch.float32) *
            (-torch.log(torch.tensor(max_freq)) / self.half_dim)
        )
        self.register_buffer('freqs', freqs)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        skip_first: bool = True,
    ) -> torch.Tensor:
        B, N, D = x.shape

        if skip_first:
            cls_token = x[:, :1, :]
            patch_tokens = x[:, 1:, :]
            rotated_patches = self._apply_rope(patch_tokens, coords)
            return torch.cat([cls_token, rotated_patches], dim=1)
        else:
            return self._apply_rope(x, coords)

    def _apply_rope(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        cx = coords[:, 0]
        cy = coords[:, 1]

        angles_x = cx.unsqueeze(-1) * self.freqs.unsqueeze(0) * 2 * torch.pi
        angles_y = cy.unsqueeze(-1) * self.freqs.unsqueeze(0) * 2 * torch.pi

        cos_x, sin_x = angles_x.cos(), angles_x.sin()
        cos_y, sin_y = angles_y.cos(), angles_y.sin()

        x = x.view(B, N, 4, self.half_dim)
        x0, x1, x2, x3 = x.unbind(dim=2)

        x0_rot = x0 * cos_x - x1 * sin_x
        x1_rot = x0 * sin_x + x1 * cos_x
        x2_rot = x2 * cos_y - x3 * sin_y
        x3_rot = x2 * sin_y + x3 * cos_y

        x_rot = torch.stack([x0_rot, x1_rot, x2_rot, x3_rot], dim=2)
        x_rot = x_rot.view(B, N, D)

        return x_rot


class MultiScalePatchEmbed(nn.Module):
    """Multi-scale patch embedding with position encoding.

    This module produces:
    1. tokens: [B, 341, D] - CLS + patch embeddings WITH position (Transformer input)
    2. targets: [B, 340, D] - Raw patch embeddings WITHOUT position (for loss)

    Key design:
    - Input patches are REAL values (not mask tokens)
    - Position encoding is added to tokens for Transformer
    - Targets have NO position (raw embeddings for prediction loss)
    - CLS token has NO position embedding
    """

    def __init__(
        self,
        dim: int = 768,
        patch_size: int = 16,
        in_channels: int = 3,
        pos_type: str = "mlp",  # "mlp" | "rope" | "none"
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size * in_channels
        self.pos_type = pos_type

        # Learnable CLS token (NO position embedding)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        # Patch projection
        self.proj = nn.Linear(self.patch_dim, dim)

        # Position encoder
        if pos_type == "mlp":
            self.pos_encoder = ContinuousPositionEncoder(dim)
        else:
            self.pos_encoder = None

        # Register center coordinates
        for _, grid_size, _ in SCALE_CONFIGS:
            centers = get_patch_centers(grid_size)
            self.register_buffer(f'centers_g{grid_size}', centers)

        all_centers = get_all_centers()
        self.register_buffer('all_centers', all_centers)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create patch tokens and targets.

        Args:
            images: [B, 3, 256, 256]

        Returns:
            tokens: [B, 341, dim] - CLS + patches with position encoding
            targets: [B, 340, dim] - Raw patch embeddings (no position, no CLS)
        """
        B = images.shape[0]
        all_patch_embeds = []
        all_pos_embeds = []

        for scale_idx, (img_size, grid_size, num_tokens) in enumerate(SCALE_CONFIGS):
            # Resize image to scale
            if img_size != 256:
                scaled = F.interpolate(
                    images,
                    size=(img_size, img_size),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                scaled = images

            # Patchify and project
            patches = patchify_single_scale(scaled, self.patch_size)
            patch_embeds = self.proj(patches)  # [B, N, dim]
            all_patch_embeds.append(patch_embeds)

            # Position encoding (for MLP type)
            if self.pos_type == "mlp":
                centers = getattr(self, f'centers_g{grid_size}')
                pos_embeds = self.pos_encoder(centers)  # [N, dim]
                all_pos_embeds.append(pos_embeds)

        # Concatenate all patch embeddings
        patch_tokens = torch.cat(all_patch_embeds, dim=1)  # [B, 340, dim]

        # Targets: raw embeddings without position (detach to prevent collapse)
        targets = patch_tokens.clone().detach()  # [B, 340, dim]

        # Add position encoding to patch tokens
        if self.pos_type == "mlp":
            all_pos = torch.cat(all_pos_embeds, dim=0)  # [340, dim]
            patch_tokens = patch_tokens + all_pos.unsqueeze(0)  # [B, 340, dim]

        # Prepend CLS token (no position)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)  # [B, 341, dim]

        return tokens, targets

    def get_coords(self) -> torch.Tensor:
        """Get all patch center coordinates (for RoPE)."""
        return self.all_centers


# Backward compatibility alias
MultiScaleTokenizer = MultiScalePatchEmbed


if __name__ == "__main__":
    print("Testing Multi-Scale Patch Embedding")
    print("=" * 60)

    images = torch.randn(2, 3, 256, 256)

    # Test with MLP position
    print("\n--- MLP Position Encoding ---")
    embed = MultiScalePatchEmbed(dim=768, pos_type="mlp")
    tokens, targets = embed(images)
    print(f"Tokens shape: {tokens.shape}")    # [2, 341, 768]
    print(f"Targets shape: {targets.shape}")  # [2, 340, 768]

    # Verify position is added to tokens but not targets
    print("\n--- Verifying Position Encoding ---")
    # Re-run without position to check
    embed_no_pos = MultiScalePatchEmbed(dim=768, pos_type="none")
    tokens_no_pos, targets_no_pos = embed_no_pos(images)

    # tokens[:, 1:] should differ from targets (position added)
    # tokens_no_pos[:, 1:] should equal targets_no_pos (no position)
    diff_with_pos = (tokens[:, 1:] - targets).abs().mean().item()
    diff_no_pos = (tokens_no_pos[:, 1:] - targets_no_pos).abs().mean().item()
    print(f"Diff with position: {diff_with_pos:.6f} (should be > 0)")
    print(f"Diff without position: {diff_no_pos:.6f} (should be ~0)")

    # Verify boundaries
    boundaries = get_scale_boundaries()
    print(f"\nScale boundaries: {boundaries}")

    # Verify parent-child mapping
    print("\n--- Parent-Child Mapping (1→4) ---")
    mappings = get_parent_child_mapping()
    for p_start, p_end, c_start, c_end in mappings:
        n_parents = p_end - p_start
        n_children = c_end - c_start
        print(f"  [{p_start}:{p_end}] ({n_parents}) → [{c_start}:{c_end}] ({n_children})")
        assert n_children == n_parents * 4, "1→4 mapping violated!"

    # Count parameters
    print(f"\nPatch embed params: {sum(p.numel() for p in embed.parameters()) / 1e6:.2f}M")
