"""
CIFAR-100 PoC for VAR-Encoder (Cross-View Prediction)

Architecture:
- Input: [CLS, S1_real, S2_real, S3_real] (ALL real embeddings!)
- Two augmented views from same image
- Cross-View Prediction:
  - view1의 S1 → view2의 S2 예측 (coarse → fine)
  - view1의 S2 → view2의 S3 예측 (coarse → fine)
  - 반대 방향도 동일하게

Attention Mask (all scales independent):
             CLS │ S1(4) │ S2(16) │ S3(64)
            ─────┼───────┼────────┼────────
       CLS │  ✓  │   ✗   │   ✗    │   ✓     ← CLS sees CLS + S3 only
       S1  │  ✓  │   ✓   │   ✗    │   ✗     ← S1 sees CLS + S1 only
       S2  │  ✓  │   ✗   │   ✓    │   ✗     ← S2 sees CLS + S2 only
       S3  │  ✓  │   ✗   │   ✗    │   ✓     ← S3 sees CLS + S3 only

Key Insight:
  - 모든 스케일이 독립적: 각 스케일은 CLS + 자기 자신만 참조
  - CLS/S3는 CLS+S3만 참조 → 인퍼런스시 CLS+S3만 필요 (65 tokens)
  - S1/S2는 훈련시에만 사용 (hierarchical loss 계산용)
  - 다른 augmentation → low-level shortcut 방지
  - Color, blur, crop이 달라 → semantic만 남음

Scales for 32x32 images (4x4 patches):
- Scale 1: 8x8 → 2x2 grid → 4 tokens
- Scale 2: 16x16 → 4x4 grid → 16 tokens
- Scale 3: 32x32 → 8x8 grid → 64 tokens
Total: 84 tokens + CLS = 85 tokens (training), 65 tokens (inference)

Usage:
    python scripts/poc_cifar.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from functools import lru_cache
import os
import random


# ============================================================
# Config
# ============================================================

@dataclass
class CIFARConfig:
    # Model
    dim: int = 256
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    patch_size: int = 4  # 4x4 patches for CIFAR
    drop: float = 0.0
    attn_drop: float = 0.0
    drop_path: float = 0.0
    temperature: float = 0.4  # InfoNCE temperature

    # Data
    batch_size: int = 256
    num_workers: int = 4

    # Training
    epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 0.005
    warmup_epochs: int = 10

    # Logging
    use_wandb: bool = True
    project: str = "vare-poc"

    # Random seed (None = no seed fixing)
    seed: Optional[int] = 41

    # Mixed precision training (AMP)
    use_amp: bool = True

    # torch.compile for PyTorch 2.0+ (can give 10-30% speedup)
    use_compile: bool = False

    # Custom attention mask (4x4 matrix for [CLS, S1, S2, S3] x [CLS, S1, S2, S3])
    # Example: [[1,0,0,1], [1,1,0,0], [1,0,1,0], [1,0,0,1]] means:
    #   - CLS sees CLS + S3
    #   - S1 sees CLS + S1
    #   - S2 sees CLS + S2
    #   - S3 sees CLS + S3
    # 1 = attend, 0 = mask out. None = use default mask.
    custom_attn_mask: Optional[List[List[int]]] = field(
        default_factory=lambda: [[1,0,0,1],
                                 [1,1,0,0],
                                 [0,1,1,0],
                                 [0,0,0,1]]
    )

    # Loss options
    use_cls_loss: bool = True  # Enable/disable CLS-CLS SimCLR loss
    use_s3_cls_loss: bool = True  # Enable/disable S3 → CLS prediction loss

    # Pooling type for spatial pooling ('mean' or 'max')
    pool_type: str = 'max'

    # Spatial matching mode for hierarchical loss
    # 'spatial': position-wise matching with flip correspondence (same position tokens)
    # 'global': all-to-all matching without correspondence (any token can match any)
    spatial_match_mode: str = 'spatial'

    # Use learnable mask tokens for S1/S2 instead of real patch embeddings
    # Similar to CLS token - lets them learn abstract representations
    use_mask_tokens: bool = True


# ============================================================
# Scale configs for CIFAR (32x32)
# ============================================================

CIFAR_SCALE_CONFIGS = [
    (8, 2, 4),     # Scale 1: 8x8 → 2x2 = 4 tokens
    (16, 4, 16),   # Scale 2: 16x16 → 4x4 = 16 tokens
    (32, 8, 64),   # Scale 3: 32x32 → 8x8 = 64 tokens
]

CIFAR_NUM_SCALES = len(CIFAR_SCALE_CONFIGS)
CIFAR_TOTAL_PATCH_TOKENS = sum(cfg[2] for cfg in CIFAR_SCALE_CONFIGS)  # 84
CIFAR_TOKENS_PER_SCALE = [1] + [cfg[2] for cfg in CIFAR_SCALE_CONFIGS]  # [1, 4, 16, 64]
CIFAR_TOTAL_TOKENS = sum(CIFAR_TOKENS_PER_SCALE)  # 85


def get_scale_boundaries() -> List[int]:
    """[0, 1, 5, 21, 85]"""
    boundaries = [0, 1]
    for _, _, num_tokens in CIFAR_SCALE_CONFIGS:
        boundaries.append(boundaries[-1] + num_tokens)
    return boundaries


def build_causal_mask_from_matrix(
    attn_matrix: List[List[int]],
    device: str = 'cpu'
) -> torch.Tensor:
    """Build 85x85 attention mask from 4x4 block attention matrix.

    Args:
        attn_matrix: 4x4 matrix where attn_matrix[i][j] = 1 means block i can attend to block j.
                     Block order: [CLS, S1, S2, S3]
                     Example: [[1,0,0,1], [1,1,0,0], [1,0,1,0], [1,0,0,1]]
        device: target device

    Returns:
        85x85 attention mask (0 = attend, -inf = mask)
    """
    total = CIFAR_TOTAL_TOKENS
    mask = torch.full((total, total), float('-inf'))

    cumsum = [0]
    for n in CIFAR_TOKENS_PER_SCALE:
        cumsum.append(cumsum[-1] + n)
    # cumsum = [0, 1, 5, 21, 85]

    # Block ranges: CLS=[0,1), S1=[1,5), S2=[5,21), S3=[21,85)
    block_ranges = [
        (cumsum[0], cumsum[1]),  # CLS: [0, 1)
        (cumsum[1], cumsum[2]),  # S1: [1, 5)
        (cumsum[2], cumsum[3]),  # S2: [5, 21)
        (cumsum[3], cumsum[4]),  # S3: [21, 85)
    ]

    for i, (row_start, row_end) in enumerate(block_ranges):
        for j, (col_start, col_end) in enumerate(block_ranges):
            if attn_matrix[i][j] == 1:
                mask[row_start:row_end, col_start:col_end] = 0

    return mask.to(device)


# Cache for default mask
_default_mask_cache: dict = {}


def build_causal_mask(device: str = 'cpu', custom_mask: Optional[List[List[int]]] = None) -> torch.Tensor:
    """Build causal mask with CLS/S3 only seeing CLS+S3.

    Mask structure (85 x 85):
             CLS │ S1(4) │ S2(16) │ S3(64)
            ─────┼───────┼────────┼────────
       CLS │  ✓  │   ✗   │   ✗    │   ✓     ← CLS sees CLS + S3 only
       S1  │  ✓  │   ✓   │   ✗    │   ✗     ← S1 sees CLS + S1 only
       S2  │  ✓  │   ✗   │   ✓    │   ✗     ← S2 sees CLS + S2 only
       S3  │  ✓  │   ✗   │   ✗    │   ✓     ← S3 sees CLS + S3 only

    Args:
        device: target device
        custom_mask: Optional 4x4 matrix to override default mask.
                     1 = attend, 0 = mask. Block order: [CLS, S1, S2, S3]

    Key Insight:
        - All scales are independent: each sees only CLS + itself
        - CLS and S3 are self-contained → inference needs only CLS + S3 (65 tokens)
        - S1, S2 are training-only for hierarchical loss
    """
    # Use custom mask if provided
    if custom_mask is not None:
        return build_causal_mask_from_matrix(custom_mask, device)

    # Use cached default mask
    if device in _default_mask_cache:
        return _default_mask_cache[device]

    # Default mask: each block sees only CLS + itself, except CLS sees CLS + S3
    default_matrix = [
        [1, 0, 0, 1],  # CLS sees CLS + S3
        [1, 1, 0, 0],  # S1 sees CLS + S1
        [1, 0, 1, 0],  # S2 sees CLS + S2
        [0, 0, 0, 1],  # S3 sees CLS + S3
    ]
    mask = build_causal_mask_from_matrix(default_matrix, device)
    _default_mask_cache[device] = mask
    return mask


@lru_cache(maxsize=1)
def build_fast_mask(device: str = 'cpu') -> torch.Tensor:
    """Build mask for fast inference (CLS + S3 only = 65 tokens).

    Mask structure (65 x 65):
             CLS │ S3(64)
            ─────┼────────
       CLS │  ✓  │   ✓     ← CLS sees CLS + S3
       S3  │  ✓  │   ✓     ← S3 sees CLS + S3

    Full bidirectional attention within CLS+S3.
    """
    total = 1 + 64  # CLS + S3
    mask = torch.zeros((total, total))  # All visible
    return mask.to(device)


# ============================================================
# Position Encoding
# ============================================================

def get_patch_centers(grid_size: int) -> torch.Tensor:
    """Compute center coordinates for each patch in [0,1]² space."""
    step = 1.0 / grid_size
    offset = step / 2

    centers = []
    for row in range(grid_size):
        for col in range(grid_size):
            x = offset + col * step
            y = offset + row * step
            centers.append([x, y])

    return torch.tensor(centers, dtype=torch.float32)


def get_all_centers_cifar() -> torch.Tensor:
    """Get all patch centers concatenated for CIFAR scales."""
    all_centers = []
    for _, grid_size, _ in CIFAR_SCALE_CONFIGS:
        centers = get_patch_centers(grid_size)
        all_centers.append(centers)
    return torch.cat(all_centers, dim=0)  # [84, 2]


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


# ============================================================
# Model Components
# ============================================================

def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """[B, C, H, W] → [B, N, patch_size^2 * C]"""
    B, C, H, W = images.shape
    num_h, num_w = H // patch_size, W // patch_size

    patches = images.reshape(B, C, num_h, patch_size, num_w, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5)
    patches = patches.reshape(B, num_h * num_w, -1)

    return patches


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0,
                 custom_attn_mask: Optional[List[List[int]]] = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = RoPE2D(self.head_dim)
        self._mask = None
        self._fast_mask = None
        self.custom_attn_mask = custom_attn_mask

    def forward(self, x: torch.Tensor, coords: torch.Tensor, fast_mode: bool = False) -> torch.Tensor:
        B, N, D = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        B, H, N, HD = q.shape
        q = q.reshape(B * H, N, HD)
        k = k.reshape(B * H, N, HD)

        q = self.rope(q, coords, skip_first=True)
        k = self.rope(k, coords, skip_first=True)

        q = q.reshape(B, H, N, HD)
        k = k.reshape(B, H, N, HD)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if fast_mode:
            # Fast inference mode: CLS + S3 only (65 tokens)
            if self._fast_mask is None or self._fast_mask.device != x.device:
                self._fast_mask = build_fast_mask(str(x.device))
            attn = attn + self._fast_mask.unsqueeze(0).unsqueeze(0)
        else:
            # Full training mode: all 85 tokens
            if self._mask is None or self._mask.device != x.device:
                self._mask = build_causal_mask(str(x.device), custom_mask=self.custom_attn_mask)
            attn = attn + self._mask.unsqueeze(0).unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 custom_attn_mask: Optional[List[List[int]]] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop, drop, custom_attn_mask=custom_attn_mask)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor, coords: torch.Tensor, fast_mode: bool = False) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), coords, fast_mode=fast_mode)
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# Strong Augmentation for Cross-View
# ============================================================

class StrongAugment(nn.Module):
    """DINO/BYOL-style strong augmentation for cross-view prediction.

    Includes both spatial and color augmentations for each view independently.
    """

    def __init__(self):
        super().__init__()
        # CIFAR mean/std
        self.mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1)

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x: torch.Tensor, shared_affine: Optional[torch.Tensor] = None,
                apply_crop: bool = False) -> Tuple[torch.Tensor, dict]:
        """Apply random augmentation to normalized images.

        Args:
            x: [B, 3, 32, 32] normalized images
            shared_affine: Optional [B, 2, 3] pre-computed affine matrix to apply
                          (for sharing affine between views)
            apply_crop: If True, apply random crop (only for view1)
        Returns:
            Tuple of:
                - [B, 3, 32, 32] augmented normalized images
                - dict with transform info: {'flip': [B] bool, 'affine': [B, 2, 3] or None}
        """
        # Ensure float32 for augmentation operations
        x = x.float()
        B, C, H, W = x.shape
        device = x.device

        # Move mean/std to device
        mean = self.mean.to(device)
        std = self.std.to(device)

        # Denormalize first
        x = x * std + mean  # [0, 1] range

        affine_theta = None  # Will store affine if applied

        # ============================================================
        # 1. SPATIAL AUGMENTATIONS
        # ============================================================

        # 1a. Random Crop (only for view1)
        if apply_crop:
            crop_ratio = 0.8
            crop_size = int(H * crop_ratio)  # 25 for CIFAR (32 * 0.8)
            # Per-sample random crop
            max_offset = H - crop_size
            top = torch.randint(0, max_offset + 1, (B,), device=device)
            left = torch.randint(0, max_offset + 1, (B,), device=device)
            # Apply crop and resize back for each sample
            cropped = []
            for i in range(B):
                crop = x[i:i+1, :, top[i]:top[i]+crop_size, left[i]:left[i]+crop_size]
                crop_resized = F.interpolate(crop, size=(H, W), mode='bilinear', align_corners=False)
                cropped.append(crop_resized)
            x = torch.cat(cropped, dim=0)

        # 1b. Random Horizontal Flip (per-view independent, tracked)
        flip_mask = torch.rand(B, device=device) < 0.5
        if flip_mask.any():
            x[flip_mask] = x[flip_mask].flip(-1)  # flip width dimension

        # 1b. Affine Transform (shared between views OR random if first call)
        if shared_affine is not None:
            # Use the shared affine from the other view
            affine_theta = shared_affine
            grid = F.affine_grid(affine_theta, x.shape, align_corners=False)
            x = F.grid_sample(x, grid, mode='bilinear',
                              padding_mode='reflection', align_corners=False)
        else:
            # First view: generate random affine (50% chance per sample)
            affine_mask = torch.rand(B, device=device) < 0.5
            if affine_mask.any():
                # Build affine for samples that need it
                B_affine = affine_mask.sum().item()

                # Random rotation angles
                angles = (torch.rand(B_affine, device=device) - 0.5) * 30  # -15 to +15 degrees
                angles_rad = angles * (torch.pi / 180.0)

                # Random scale
                scales = 0.9 + torch.rand(B_affine, device=device) * 0.2  # 0.9 to 1.1

                # Random translation (in normalized coords, -1 to 1)
                tx = (torch.rand(B_affine, device=device) - 0.5) * 0.2  # -0.1 to 0.1
                ty = (torch.rand(B_affine, device=device) - 0.5) * 0.2

                # Build affine matrix [B_affine, 2, 3]
                cos_a = torch.cos(angles_rad)
                sin_a = torch.sin(angles_rad)

                theta_partial = torch.zeros(B_affine, 2, 3, device=device)
                theta_partial[:, 0, 0] = cos_a * scales
                theta_partial[:, 0, 1] = -sin_a * scales
                theta_partial[:, 0, 2] = tx
                theta_partial[:, 1, 0] = sin_a * scales
                theta_partial[:, 1, 1] = cos_a * scales
                theta_partial[:, 1, 2] = ty

                # Build full theta for all samples (identity for non-affine)
                affine_theta = torch.zeros(B, 2, 3, device=device)
                affine_theta[:, 0, 0] = 1.0
                affine_theta[:, 1, 1] = 1.0
                affine_theta[affine_mask] = theta_partial

                # Apply affine transform to all (identity for non-affine samples)
                grid = F.affine_grid(affine_theta, x.shape, align_corners=False)
                x = F.grid_sample(x, grid, mode='bilinear',
                                  padding_mode='reflection', align_corners=False)

        transform_info = {
            'flip': flip_mask,  # [B] bool - only flip varies between views
            'affine': affine_theta,  # [B, 2, 3] or None - shared between views
        }

        # ============================================================
        # 2. COLOR AUGMENTATIONS
        # ============================================================

        # 2a. Color Jitter (per-sample random)
        # Brightness
        brightness = 1.0 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.8  # [0.6, 1.4]
        x = x * brightness

        # Contrast
        gray = x.mean(dim=1, keepdim=True)
        contrast = 1.0 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.8  # [0.6, 1.4]
        x = gray + (x - gray) * contrast

        # Saturation
        gray = x.mean(dim=1, keepdim=True)
        saturation = 1.0 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.8  # [0.6, 1.4]
        x = gray + (x - gray) * saturation

        # Hue shift (simplified - just rotate channels slightly)
        hue_mask = torch.rand(B, device=device) < 0.5
        if hue_mask.any():
            x[hue_mask] = x[hue_mask][:, [2, 1, 0], :, :]

        # 2b. Random Grayscale (20% chance)
        gray_mask = torch.rand(B, device=device) < 0.2
        if gray_mask.any():
            gray_val = x[gray_mask].mean(dim=1, keepdim=True)
            x[gray_mask] = gray_val.expand(-1, 3, -1, -1)

        # 2c. Gaussian Blur (50% chance) - simple 3x3 averaging
        blur_mask = torch.rand(B, device=device) < 0.5
        if blur_mask.any():
            kernel = torch.ones(3, 1, 3, 3, device=device) / 9.0
            blurred = F.conv2d(
                F.pad(x[blur_mask], (1, 1, 1, 1), mode='replicate'),
                kernel,
                groups=3
            )
            x[blur_mask] = blurred

        # 2d. Random Solarize (10% chance)
        solar_mask = torch.rand(B, device=device) < 0.1
        if solar_mask.any():
            threshold = 0.5
            inverted = 1.0 - x[solar_mask]
            high_mask = x[solar_mask] > threshold
            x[solar_mask] = torch.where(high_mask, inverted, x[solar_mask])

        # Clamp and renormalize
        x = x.clamp(0, 1)
        x = (x - mean) / std

        return x, transform_info


class VAREncoderCIFAR(nn.Module):
    """VAR-Encoder for CIFAR-100 with Cross-View Prediction.

    Architecture:
        Input: [CLS, S1_real, S2_real, S3_real] (ALL real embeddings!)
        Two augmented views from same image
        Cross-View Prediction:
            - view1의 S1 → view2의 S2 예측 (coarse → fine)
            - view1의 S2 → view2의 S3 예측 (coarse → fine)
            - 반대 방향도 동일하게

    Key insight:
        - 다른 augmentation → low-level shortcut 방지
        - Color, blur, crop이 달라 → semantic만 남음
        - DINO/BYOL과 유사한 검증된 방법
    """

    def __init__(self, config: CIFARConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.patch_dim = config.patch_size ** 2 * 3

        # Patch projection
        self.proj = nn.Linear(self.patch_dim, config.dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dim))

        # Learnable mask tokens for S1/S2 (optional, like CLS)
        if config.use_mask_tokens:
            self.s1_mask_token = nn.Parameter(torch.zeros(1, 1, config.dim))  # Single token, broadcast to 4
            self.s2_mask_token = nn.Parameter(torch.zeros(1, 1, config.dim))  # Single token, broadcast to 16

        # Patch coordinates for RoPE
        coords = get_all_centers_cifar()  # [84, 2]
        self.register_buffer('coords', coords)

        # S3-only coordinates for fast inference (8x8 = 64 tokens)
        coords_s3 = get_patch_centers(8)  # [64, 2]
        self.register_buffer('coords_s3', coords_s3)

        # Boundaries: [0, 1, 5, 21, 85]
        self.boundaries = get_scale_boundaries()

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(config.dim, config.num_heads, config.mlp_ratio,
                  config.drop, config.attn_drop,
                  custom_attn_mask=config.custom_attn_mask)
            for _ in range(config.depth)
        ])
        self.norm = nn.LayerNorm(config.dim)

        # Cross-view predictor (BYOL-style asymmetry)
        self.predictor = nn.Sequential(
            nn.Linear(config.dim, config.dim * 2),
            nn.GELU(),
            nn.Linear(config.dim * 2, config.dim),
        )

        # Strong augmentation for cross-view
        self.augment = StrongAugment()

        self.dim = config.dim

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        if self.config.use_mask_tokens:
            nn.init.trunc_normal_(self.s1_mask_token, std=0.02)
            nn.init.trunc_normal_(self.s2_mask_token, std=0.02)

    def encode_all_scales(self, images: torch.Tensor) -> dict:
        """Encode images and return hidden states for all scales.

        Args:
            images: [B, 3, 32, 32]
        Returns:
            dict with 'cls', 's1', 's2', 's3' hidden states
        """
        B = images.shape[0]

        # Get real patch embeddings for each scale
        all_embeds = []
        for img_size, grid_size, num_tokens in CIFAR_SCALE_CONFIGS:
            if img_size != 32:
                scaled = F.interpolate(images, size=(img_size, img_size),
                                       mode='bilinear', align_corners=False)
            else:
                scaled = images

            patches = patchify(scaled, self.patch_size)  # [B, N, patch_dim]
            embeds = self.proj(patches)  # [B, N, dim]
            all_embeds.append(embeds)

        s1_real, s2_real, s3_real = all_embeds

        # Use mask tokens or real embeddings for S1/S2
        if self.config.use_mask_tokens:
            # Learnable mask tokens (like CLS) instead of real patches
            s1_tokens = self.s1_mask_token.expand(B, 4, -1)   # [B, 4, dim]
            s2_tokens = self.s2_mask_token.expand(B, 16, -1)  # [B, 16, dim]
        else:
            # Real patch embeddings
            s1_tokens = s1_real
            s2_tokens = s2_real

        # Construct input: [CLS, S1, S2, S3_real]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, dim]
        tokens = torch.cat([cls_tokens, s1_tokens, s2_tokens, s3_real], dim=1)  # [B, 85, dim]

        # Transformer with causal mask
        for block in self.blocks:
            tokens = block(tokens, self.coords)
        hidden = self.norm(tokens)

        # Extract hidden states for each scale
        return {
            'cls': hidden[:, 0],           # [B, dim]
            's1': hidden[:, 1:5],          # [B, 4, dim]
            's2': hidden[:, 5:21],         # [B, 16, dim]
            's3': hidden[:, 21:85],        # [B, 64, dim]
        }

    def spatial_pool(self, x: torch.Tensor, from_grid: int, to_grid: int) -> torch.Tensor:
        """Spatially downsample tokens from from_grid to to_grid.

        Args:
            x: [B, from_grid^2, dim]
            from_grid: source grid size (e.g., 4 for S2, 8 for S3)
            to_grid: target grid size (e.g., 2 for S1, 4 for S2)
        Returns:
            [B, to_grid^2, dim]
        """
        B, N, D = x.shape
        assert N == from_grid * from_grid

        # Reshape to 2D grid
        x = x.view(B, from_grid, from_grid, D).permute(0, 3, 1, 2)  # [B, D, H, W]

        # Pool down using configured pool type
        pool_size = from_grid // to_grid
        if self.config.pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size, stride=pool_size)
        else:  # mean
            x = F.avg_pool2d(x, kernel_size=pool_size, stride=pool_size)  # [B, D, to_grid, to_grid]

        # Reshape back
        x = x.permute(0, 2, 3, 1).view(B, to_grid * to_grid, D)  # [B, to_grid^2, D]
        return x

    def global_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Pool all tokens to a single vector.

        Args:
            x: [B, N, dim] - spatial tokens
        Returns:
            [B, dim] - pooled representation
        """
        if self.config.pool_type == 'max':
            return x.max(dim=1)[0]
        else:  # mean
            return x.mean(dim=1)

    def compute_flip_correspondence(
        self,
        flip1: torch.Tensor,
        flip2: torch.Tensor,
        grid_size: int,
    ) -> torch.Tensor:
        """Compute token correspondence based on flip difference between views.

        Since affine is shared, only flip differs between views.
        If both views have same flip status → identity mapping
        If flip status differs → horizontal flip mapping

        Args:
            flip1: [B] bool - whether view1 was flipped
            flip2: [B] bool - whether view2 was flipped
            grid_size: grid size (2 for S1, 4 for S2, 8 for S3)

        Returns:
            correspondence: [B, N] indices mapping view1 tokens to view2 tokens
        """
        B = flip1.shape[0]
        device = flip1.device
        N = grid_size * grid_size

        # Create identity mapping [0, 1, 2, ..., N-1]
        identity = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # [B, N]

        # Create horizontal flip mapping
        # For grid_size=2: [0,1,2,3] -> [1,0,3,2] (flip columns within each row)
        # For grid_size=4: [0,1,2,3,4,5,...] -> [3,2,1,0,7,6,5,4,...]
        flip_map = torch.arange(N, device=device).view(grid_size, grid_size)
        flip_map = flip_map.flip(1).flatten()  # Flip columns
        flip_map = flip_map.unsqueeze(0).expand(B, -1)  # [B, N]

        # XOR: need to flip correspondence only if exactly one view was flipped
        need_flip = flip1 ^ flip2  # [B] bool

        # Select between identity and flip mapping
        correspondence = torch.where(
            need_flip.unsqueeze(1).expand(-1, N),
            flip_map,
            identity
        )  # [B, N]

        return correspondence

    def forward(self, images: torch.Tensor) -> dict:
        """
        Args:
            images: [B, 3, 32, 32]

        Returns:
            dict with 'loss', 'cls'
        """
        B = images.shape[0]

        # Two augmented views from same image
        # - Random crop is applied only to view1
        # - Affine is shared (applied identically to both views)
        # - Flip is independent (different per view, tracked for correspondence)
        view1, t1 = self.augment(images, apply_crop=True)  # First view: crop + generates affine
        view2, t2 = self.augment(images, shared_affine=t1['affine'])  # Second view: no crop, reuses affine

        # Encode both views (shared weights)
        h1 = self.encode_all_scales(view1)  # dict: {cls, s1, s2, s3}
        h2 = self.encode_all_scales(view2)  # dict: {cls, s1, s2, s3}

        # ============================================================
        # 1. CLS-CLS SimCLR Loss (global representation - spatial invariance)
        # ============================================================
        if self.config.use_cls_loss:
            loss_cls = self.infonce_cls(h1['cls'], h2['cls'])
        else:
            loss_cls = torch.tensor(0.0, device=images.device)

        # ============================================================
        # 2. Cross-view Hierarchical Loss
        # ============================================================
        # S1 (2x2=4) predicts S2 (4x4=16) pooled to (2x2=4)
        # S2 (4x4=16) predicts S3 (8x8=64) pooled to (4x4=16)

        h2_s2_pooled = self.spatial_pool(h2['s2'], from_grid=4, to_grid=2)  # [B, 4, dim]
        h2_s3_pooled = self.spatial_pool(h2['s3'], from_grid=8, to_grid=4)  # [B, 16, dim]
        h1_s2_pooled = self.spatial_pool(h1['s2'], from_grid=4, to_grid=2)  # [B, 4, dim]
        h1_s3_pooled = self.spatial_pool(h1['s3'], from_grid=8, to_grid=4)  # [B, 16, dim]

        if self.config.spatial_match_mode == 'spatial':
            # Spatial matching: position-wise with flip correspondence
            corr_s1 = self.compute_flip_correspondence(t1['flip'], t2['flip'], grid_size=2)
            corr_s2 = self.compute_flip_correspondence(t1['flip'], t2['flip'], grid_size=4)

            loss_12_s1s2 = self.infonce_spatial_aligned(h1['s1'], h2_s2_pooled, corr_s1)
            loss_12_s2s3 = self.infonce_spatial_aligned(h1['s2'], h2_s3_pooled, corr_s2)
            loss_21_s1s2 = self.infonce_spatial_aligned(h2['s1'], h1_s2_pooled, corr_s1)
            loss_21_s2s3 = self.infonce_spatial_aligned(h2['s2'], h1_s3_pooled, corr_s2)
        else:
            # Global matching: all-to-all without correspondence
            loss_12_s1s2 = self.infonce_spatial(h1['s1'], h2_s2_pooled)
            loss_12_s2s3 = self.infonce_spatial(h1['s2'], h2_s3_pooled)
            loss_21_s1s2 = self.infonce_spatial(h2['s1'], h1_s2_pooled)
            loss_21_s2s3 = self.infonce_spatial(h2['s2'], h1_s3_pooled)

        # loss_hier = (loss_12_s1s2 + loss_12_s2s3 + loss_21_s1s2 + loss_21_s2s3) / 4
        loss_hier = (loss_12_s1s2 + loss_12_s2s3) / 2

        # ============================================================
        # 3. S3 → CLS Prediction Loss (fine → global)
        # ============================================================
        if self.config.use_s3_cls_loss:
            # Pool S3 to single vector and predict the other view's CLS
            h1_s3_global = self.global_pool(h1['s3'])  # [B, dim]
            h2_s3_global = self.global_pool(h2['s3'])  # [B, dim]
            # Cross-view: S3 from view1 → CLS from view2, and vice versa
            # loss_s3_cls = (self.infonce_cls(h1_s3_global, h2['cls']) +
            #                self.infonce_cls(h2_s3_global, h1['cls'])) / 2
            loss_s3_cls = self.infonce_cls(h1_s3_global, h2['cls'])
        else:
            loss_s3_cls = torch.tensor(0.0, device=images.device)

        # ============================================================
        # 4. Combined Loss
        # ============================================================
        # Count active losses for weighting
        active_losses = []
        if self.config.use_cls_loss:
            active_losses.append(loss_cls)
        active_losses.append(loss_hier)  # Always active
        if self.config.use_s3_cls_loss:
            active_losses.append(loss_s3_cls)

        # Equal weight for all active losses
        loss = sum(active_losses) / len(active_losses)

        return {
            'loss': loss,
            'loss_cls': loss_cls.item() if self.config.use_cls_loss else 0.0,
            'loss_hier': loss_hier.item(),
            'loss_s3_cls': loss_s3_cls.item() if self.config.use_s3_cls_loss else 0.0,
            'cls': h1['cls'],
        }

    def infonce_cls(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """SimCLR-style InfoNCE for CLS tokens (symmetric).

        Args:
            z1: [B, dim] - CLS from view1
            z2: [B, dim] - CLS from view2
        Returns:
            scalar loss
        """
        B = z1.shape[0]
        tau = self.config.temperature

        # Project both
        z1_proj = self.predictor(z1)
        z2_proj = self.predictor(z2)

        z1_norm = F.normalize(z1_proj, dim=-1)
        z2_norm = F.normalize(z2_proj, dim=-1)

        # Symmetric InfoNCE (SimCLR style)
        # z1 → z2
        logits_12 = z1_norm @ z2_norm.T / tau  # [B, B]
        labels = torch.arange(B, device=logits_12.device)
        loss_12 = F.cross_entropy(logits_12, labels)

        # z2 → z1
        logits_21 = z2_norm @ z1_norm.T / tau
        loss_21 = F.cross_entropy(logits_21, labels)

        return (loss_12 + loss_21) / 2

    def infonce_spatial(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Spatial-aware InfoNCE: position-wise matching.

        Args:
            pred: [B, N, dim] - source tokens (e.g., S1 with N=4)
            target: [B, N, dim] - target tokens (pooled, same N)
        Returns:
            scalar loss
        """
        B, N, D = pred.shape
        tau = self.config.temperature

        # Flatten to [B*N, dim] for per-position matching
        pred_flat = pred.reshape(B * N, D)
        target_flat = target.reshape(B * N, D)

        # Project pred only (asymmetry)
        pred_proj = self.predictor(pred_flat)
        pred_norm = F.normalize(pred_proj, dim=-1)
        target_norm = F.normalize(target_flat.detach(), dim=-1)

        # InfoNCE: each position matches its corresponding position
        # logits[i, j] = similarity between pred[i] and target[j]
        logits = pred_norm @ target_norm.T / tau  # [B*N, B*N]
        labels = torch.arange(B * N, device=logits.device)

        return F.cross_entropy(logits, labels)

    def infonce_spatial_aligned(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        correspondence: torch.Tensor,
    ) -> torch.Tensor:
        """Spatial InfoNCE with flip-aware correspondence.

        Instead of assuming pred[i] matches target[i], we use the computed
        correspondence to match tokens that represent the same spatial location
        in the original image (accounting for flip differences).

        Args:
            pred: [B, N, dim] - source tokens from view1
            target: [B, N, dim] - target tokens from view2 (pooled)
            correspondence: [B, N] - for each pred token, which target token it corresponds to
        Returns:
            scalar loss
        """
        B, N, D = pred.shape
        tau = self.config.temperature
        device = pred.device

        # Gather corresponding targets using correspondence mapping
        # target_aligned[b, n] = target[b, correspondence[b, n]]
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)  # [B, N]
        target_aligned = target[batch_indices, correspondence]  # [B, N, dim]

        # Flatten for InfoNCE
        pred_flat = pred.reshape(B * N, D)  # [B*N, dim]
        target_flat = target_aligned.reshape(B * N, D)  # [B*N, dim]

        # Project pred only (asymmetry)
        pred_proj = self.predictor(pred_flat)
        pred_norm = F.normalize(pred_proj, dim=-1)
        target_norm = F.normalize(target_flat.detach(), dim=-1)

        # InfoNCE: each position matches its corresponding position
        logits = pred_norm @ target_norm.T / tau  # [B*N, B*N]
        labels = torch.arange(B * N, device=device)

        return F.cross_entropy(logits, labels)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Extract CLS representation (for evaluation, uses full 85 tokens)."""
        with torch.no_grad():
            h = self.encode_all_scales(images)
        return h['cls']

    def encode_fast(self, images: torch.Tensor) -> torch.Tensor:
        """Fast inference using only CLS + S3 (65 tokens instead of 85).

        Since CLS and S3 only attend to CLS + S3, we can skip S1/S2 entirely.
        This is 23% fewer tokens → faster inference.

        Args:
            images: [B, 3, 32, 32]
        Returns:
            [B, dim] CLS representation
        """
        B = images.shape[0]

        # Only get S3 patches (full resolution 32x32 → 8x8 grid)
        patches = patchify(images, self.patch_size)  # [B, 64, patch_dim]
        s3_embed = self.proj(patches)  # [B, 64, dim]

        # Construct input: [CLS, S3] only (65 tokens)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, dim]
        tokens = torch.cat([cls_tokens, s3_embed], dim=1)  # [B, 65, dim]

        # Transformer with fast mask (CLS + S3 only)
        for block in self.blocks:
            tokens = block(tokens, self.coords_s3, fast_mode=True)
        hidden = self.norm(tokens)

        return hidden[:, 0]  # [B, dim] CLS token


# ============================================================
# Training
# ============================================================

def get_cifar100_loaders(config: CIFARConfig):
    # No augmentation here - StrongAugment handles everything per-view
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_dataset = datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=2 if config.num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=2 if config.num_workers > 0 else None,
    )

    return train_loader, test_loader


def train_epoch(model, loader, optimizer, device, epoch, base_seed: Optional[int] = None,
                scaler: Optional[torch.amp.GradScaler] = None, use_amp: bool = False):
    model.train()
    total_loss = 0
    total_loss_cls = 0
    total_loss_hier = 0
    total_loss_s3_cls = 0

    # Set epoch-specific seed for reproducibility with diversity
    if base_seed is not None:
        epoch_seed = base_seed + epoch
        torch.manual_seed(epoch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(epoch_seed)

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for images, _ in pbar:
        images = images.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

        # Mixed precision training
        with torch.amp.autocast('cuda', enabled=use_amp):
            output = model(images)
            loss = output['loss']

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_loss_cls += output['loss_cls']
        total_loss_hier += output['loss_hier']
        total_loss_s3_cls += output['loss_s3_cls']
        pbar.set_postfix(
            loss=f"{loss.item():.3f}",
            cls=f"{output['loss_cls']:.3f}",
            hier=f"{output['loss_hier']:.3f}",
            s3c=f"{output['loss_s3_cls']:.3f}"
        )

    n = len(loader)
    return {
        'loss': total_loss / n,
        'loss_cls': total_loss_cls / n,
        'loss_hier': total_loss_hier / n,
        'loss_s3_cls': total_loss_s3_cls / n,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0

    for images, _ in loader:
        images = images.to(device)
        output = model(images)
        total_loss += output['loss'].item()

    return total_loss / len(loader)


# ============================================================
# DINO-style Evaluation
# ============================================================

@torch.no_grad()
def extract_features(model, loader, device):
    """Extract CLS features and labels from dataset."""
    model.eval()
    features = []
    labels = []

    for images, targets in tqdm(loader, desc="Extracting features"):
        images = images.to(device)
        feat = model.encode(images)
        features.append(feat.cpu())
        labels.append(targets)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    return features.numpy(), labels.numpy()


def knn_evaluate(model, train_loader, test_loader, device, k=5):
    """k-NN classification accuracy (DINO-style)."""
    print(f"\n--- k-NN Evaluation (k={k}) ---")

    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)

    # Normalize
    train_features = train_features / np.linalg.norm(train_features, axis=1, keepdims=True)
    test_features = test_features / np.linalg.norm(test_features, axis=1, keepdims=True)

    # k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(train_features, train_labels)

    accuracy = knn.score(test_features, test_labels)
    print(f"k-NN Top-1 Accuracy: {accuracy * 100:.2f}%")

    return accuracy


def visualize_tsne(model, test_loader, device, save_path='tsne.png', n_samples=2000):
    """t-SNE visualization of CLS embeddings."""
    print("\n--- t-SNE Visualization ---")

    model.eval()
    features = []
    labels = []
    count = 0

    with torch.no_grad():
        for images, targets in test_loader:
            if count >= n_samples:
                break
            images = images.to(device)
            feat = model.encode(images)
            features.append(feat.cpu())
            labels.append(targets)
            count += len(images)

    features = torch.cat(features, dim=0)[:n_samples].numpy()
    labels = torch.cat(labels, dim=0)[:n_samples].numpy()

    print(f"Running t-SNE on {len(features)} samples...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings = tsne.fit_transform(features)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1],
                          c=labels, cmap='tab20', s=5, alpha=0.7)
    plt.colorbar(scatter)
    plt.title('t-SNE of CLS Embeddings (Cross-View Prediction)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"t-SNE saved to {save_path}")


@torch.no_grad()
def visualize_attention(model, test_loader, device, save_path='attention.png', n_images=4):
    """DINO-style attention visualization with per-head breakdown.

    Layout:
        Row 0: Original images
        Row 1: Head 0 attention
        Row 2: Head 1 attention
        ...
        Row H: Head H-1 attention
        Row H+1: Mean attention (all heads)
    """
    model.eval()

    images, labels = next(iter(test_loader))
    images = images[:n_images].to(device)

    # Denormalize
    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1).to(device)
    images_viz = (images * std + mean).clamp(0, 1).cpu()

    # Get attention from last layer
    B = images.shape[0]

    # Build input: [CLS, S1_real, S2_real, S3_real]
    all_embeds = []
    for img_size, grid_size, num_tokens in CIFAR_SCALE_CONFIGS:
        if img_size != 32:
            scaled = F.interpolate(images, size=(img_size, img_size),
                                   mode='bilinear', align_corners=False)
        else:
            scaled = images
        patches = patchify(scaled, model.patch_size)
        embeds = model.proj(patches)
        all_embeds.append(embeds)

    s1_real, s2_real, s3_real = all_embeds
    cls_tokens = model.cls_token.expand(B, -1, -1)
    tokens = torch.cat([cls_tokens, s1_real, s2_real, s3_real], dim=1)

    # Forward through blocks, extract last attention
    for block in model.blocks[:-1]:
        tokens = block(tokens, model.coords)

    # Last block - extract attention
    last_block = model.blocks[-1]
    x = last_block.norm1(tokens)
    B_cur, N, D = x.shape
    num_heads = last_block.attn.num_heads
    qkv = last_block.attn.qkv(x).reshape(B_cur, N, 3, num_heads, last_block.attn.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    B_attn, H, N, HD = q.shape
    q = q.reshape(B_attn * H, N, HD)
    k = k.reshape(B_attn * H, N, HD)
    q = last_block.attn.rope(q, model.coords, skip_first=True)
    k = last_block.attn.rope(k, model.coords, skip_first=True)
    q = q.reshape(B_attn, H, N, HD)
    k = k.reshape(B_attn, H, N, HD)

    attn = (q @ k.transpose(-2, -1)) * last_block.attn.scale
    mask = build_causal_mask(str(device))
    attn = attn + mask.unsqueeze(0).unsqueeze(0)
    attn = attn.softmax(dim=-1)  # [B, H, N, N]
    attn = attn.cpu()

    # Plot: Original + each head + mean
    # Rows: Original, Head0, Head1, ..., HeadH-1, Mean
    n_rows = 1 + num_heads + 1  # original + heads + mean
    fig, axes = plt.subplots(n_rows, n_images, figsize=(n_images * 2.5, n_rows * 2.2))

    for i in range(n_images):
        # Row 0: Original image
        axes[0, i].imshow(images_viz[i].permute(1, 2, 0))
        axes[0, i].set_title(f'Class {labels[i].item()}', fontsize=9)
        axes[0, i].axis('off')

        # Row 1 to H: Per-head attention
        for h in range(num_heads):
            # CLS attention to S3 (positions 21:85 → 8x8)
            cls_attn = attn[i, h, 0, 21:85].reshape(8, 8).numpy()
            cls_attn = np.kron(cls_attn, np.ones((4, 4)))  # Upsample to 32x32

            axes[1 + h, i].imshow(images_viz[i].permute(1, 2, 0))
            axes[1 + h, i].imshow(cls_attn, cmap='hot', alpha=0.6)
            axes[1 + h, i].axis('off')

        # Last row: Mean attention
        cls_attn_mean = attn[i, :, 0, 21:85].mean(dim=0).reshape(8, 8).numpy()
        cls_attn_mean = np.kron(cls_attn_mean, np.ones((4, 4)))

        axes[-1, i].imshow(images_viz[i].permute(1, 2, 0))
        axes[-1, i].imshow(cls_attn_mean, cmap='hot', alpha=0.6)
        axes[-1, i].axis('off')

    # Row labels
    axes[0, 0].set_ylabel('Original', fontsize=10)
    for h in range(num_heads):
        axes[1 + h, 0].set_ylabel(f'Head {h}', fontsize=10)
    axes[-1, 0].set_ylabel('Mean', fontsize=10)

    plt.suptitle('CLS → S3 Attention per Head (Last Layer)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Attention saved to {save_path}")


def full_evaluation(model, train_loader, test_loader, device, output_dir='outputs'):
    """Run DINO-style evaluation: k-NN + t-SNE + Attention."""
    os.makedirs(output_dir, exist_ok=True)

    knn_acc = knn_evaluate(model, train_loader, test_loader, device)
    visualize_tsne(model, test_loader, device,
                   save_path=os.path.join(output_dir, 'tsne.png'))
    visualize_attention(model, test_loader, device,
                        save_path=os.path.join(output_dir, 'attention.png'))

    return knn_acc


# ============================================================
# Seed fixing
# ============================================================

def set_seed(seed: int, deterministic: bool = False):
    """Fix random seed for reproducibility.

    Args:
        seed: Random seed
        deterministic: If True, use deterministic algorithms (slower but reproducible).
                      If False, allow non-deterministic for speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Allow cudnn to find optimal algorithms (faster)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    print(f"Random seed fixed: {seed} (deterministic={deterministic})")


# ============================================================
# Main
# ============================================================

def main():
    config = CIFARConfig()

    # Fix random seed if specified
    if config.seed is not None:
        set_seed(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("VAR-Encoder CIFAR-100 PoC (Cross-View Prediction)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Scales: {CIFAR_SCALE_CONFIGS}")
    print(f"Total tokens: {CIFAR_TOTAL_TOKENS} (CLS + {CIFAR_TOTAL_PATCH_TOKENS} patches)")

    boundaries = get_scale_boundaries()
    print(f"\nArchitecture:")
    print(f"  Input: [CLS, S1_real, S2_real, S3_real] (ALL real embeddings!)")
    print(f"  Attention Mask (all scales independent):")
    print(f"    CLS [{boundaries[0]}:{boundaries[1]}] → sees CLS + S3 only")
    print(f"    S1 [{boundaries[1]}:{boundaries[2]}] → sees CLS + S1 only")
    print(f"    S2 [{boundaries[2]}:{boundaries[3]}] → sees CLS + S2 only")
    print(f"    S3 [{boundaries[3]}:{boundaries[4]}] → sees CLS + S3 only")
    print(f"  Fast Inference:")
    print(f"    CLS + S3 only (65 tokens instead of 85)")
    print(f"    S1, S2 are training-only for hierarchical loss")
    print(f"  Cross-View Prediction:")
    print(f"    view1 = augment(image)")
    print(f"    view2 = augment(image)  # different random augmentation")
    print(f"    h1_S1 → h2_S2 (InfoNCE)  # v1 coarse → v2 fine")
    print(f"    h1_S2 → h2_S3 (InfoNCE)")
    print(f"    h2_S1 → h1_S2 (InfoNCE)  # reverse direction")
    print(f"    h2_S2 → h1_S3 (InfoNCE)")
    print(f"  Temperature: {config.temperature}")
    if config.custom_attn_mask is not None:
        print(f"\n  Custom Attention Mask (4x4):")
        for i, row in enumerate(config.custom_attn_mask):
            block_name = ['CLS', 'S1 ', 'S2 ', 'S3 '][i]
            print(f"    {block_name}: {row}")
    if config.seed is not None:
        print(f"\n  Random Seed: {config.seed}")

    # Model
    model = VAREncoderCIFAR(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {num_params / 1e6:.2f}M")

    # torch.compile (PyTorch 2.0+)
    if config.use_compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')
        print("Model compiled!")

    # Data
    train_loader, test_loader = get_cifar100_loaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp) if config.use_amp else None

    # Wandb
    if config.use_wandb:
        wandb.init(project=config.project, config=vars(config))
        wandb.watch(model)

    # Training loop
    print("\n" + "=" * 60)
    print(f"Starting training... (AMP={'ON' if config.use_amp else 'OFF'})")
    print("=" * 60)

    best_loss = float('inf')
    best_knn = 0.0
    last_knn = 0.0

    for epoch in range(1, config.epochs + 1):
        train_stats = train_epoch(
            model, train_loader, optimizer, device, epoch,
            base_seed=config.seed, scaler=scaler, use_amp=config.use_amp
        )
        test_loss = evaluate(model, test_loader, device)
        scheduler.step()

        # k-NN eval every 10 epochs
        knn_evaluated = False
        if epoch % 10 == 0 or epoch == 1:
            last_knn = knn_evaluate(model, train_loader, test_loader, device)
            knn_evaluated = True
            if last_knn > best_knn:
                best_knn = last_knn

            # Visualizations
            epoch_dir = f'outputs/epoch{epoch}'
            os.makedirs(epoch_dir, exist_ok=True)
            visualize_tsne(model, test_loader, device,
                           save_path=f'{epoch_dir}/tsne.png')
            visualize_attention(model, test_loader, device,
                                save_path=f'{epoch_dir}/attention.png')

            if config.use_wandb:
                wandb.log({
                    'viz/tsne': wandb.Image(f'{epoch_dir}/tsne.png'),
                    'viz/attention': wandb.Image(f'{epoch_dir}/attention.png'),
                })

        knn_str = f"knn={last_knn*100:.2f}%" if knn_evaluated else f"knn={last_knn*100:.2f}% (cached)"
        print(f"Epoch {epoch}: loss={train_stats['loss']:.4f} (cls={train_stats['loss_cls']:.3f}, hier={train_stats['loss_hier']:.3f}, s3c={train_stats['loss_s3_cls']:.3f}), test={test_loss:.4f}, {knn_str}")

        if config.use_wandb:
            log_dict = {
                'train/loss': train_stats['loss'],
                'train/loss_cls': train_stats['loss_cls'],
                'train/loss_hier': train_stats['loss_hier'],
                'train/loss_s3_cls': train_stats['loss_s3_cls'],
                'test/loss': test_loss,
                'lr': scheduler.get_last_lr()[0],
            }
            if knn_evaluated:
                log_dict['eval/knn_acc'] = last_knn
            wandb.log(log_dict)

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), 'best_model_cifar.pt')
            print(f"  → Best model saved! (loss={best_loss:.4f})")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    model.load_state_dict(torch.load('best_model_cifar.pt'))
    final_knn = full_evaluation(model, train_loader, test_loader, device)

    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Best test loss: {best_loss:.4f}")
    print(f"Best k-NN accuracy: {best_knn * 100:.2f}%")
    print(f"Final k-NN accuracy: {final_knn * 100:.2f}%")
    print("=" * 60)

    if config.use_wandb:
        wandb.log({
            'final/knn_acc': final_knn,
            'final/best_loss': best_loss,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
