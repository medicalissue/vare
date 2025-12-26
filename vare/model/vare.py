"""VAR-Encoder: Coarse-to-Fine Autoregressive Vision Encoder.

Main model that learns image representations by predicting
finer-scale patches from coarser-scale context.

Key Design:
- Input: CLS + real patch embeddings (with position encoding)
- Hidden: Latent representations (this is what we use for downstream)
- Prediction: Hidden → MLP(1→4) → predict NEXT scale's GT
- Inference: Use CLS hidden state as representation, discard prediction head

Token structure:
- [CLS(1), S1(4), S2(16), S3(64), S4(256)] = 341 tokens

Attention pattern:
- CLS: Sees ALL tokens (global aggregation for representation)
- S1: CLS + S1 (bidirectional within S1)
- S2: CLS + S1 + S2 (bidirectional within S2)
- S3: CLS + S1 + S2 + S3 (bidirectional within S3)
- S4: All tokens (bidirectional within S4)

Prediction mapping (auxiliary task):
- h_S1 (4) → predict S2 GT (16 tokens)
- h_S2 (16) → predict S3 GT (64 tokens)
- h_S3 (64) → predict S4 GT (256 tokens)
- CLS, S4: No prediction (CLS aggregates, S4 is finest)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Literal
from dataclasses import dataclass

from .patchify import (
    MultiScalePatchEmbed,
    SCALE_CONFIGS,
    NUM_SCALES,
    TOTAL_PATCH_TOKENS,
    get_scale_boundaries,
    get_parent_child_mapping,
)
from .transformer import TransformerEncoder
from .attention import TOKENS_PER_SCALE, TOTAL_TOKENS


@dataclass
class VAREncoderConfig:
    """Configuration for VAR-Encoder."""
    dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    patch_size: int = 16
    drop: float = 0.0
    attn_drop: float = 0.0
    drop_path: float = 0.1
    loss_type: str = "mse"  # "mse" or "cosine"
    pos_type: str = "mlp"   # "mlp" or "rope" or "none"
    use_flash_attn: bool = True


class PredictionHead(nn.Module):
    """1→4 Prediction Head.

    Each parent token predicts 4 child tokens in the next scale.
    Input: [B, N, D] parent hidden states
    Output: [B, N*4, D] predicted child tokens
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim * 4),  # 1 → 4 outputs
        )
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] parent hidden states

        Returns:
            out: [B, N*4, D] predicted child tokens (flattened)
        """
        B, N, D = x.shape
        out = self.net(x)  # [B, N, D*4]
        out = out.reshape(B, N * 4, D)  # [B, N*4, D]
        return out


class VAREncoder(nn.Module):
    """VAR-Encoder: Coarse-to-Fine Autoregressive Vision Encoder.

    Architecture:
        1. Multi-scale patchification: real patch embeddings + position
        2. Transformer with scale-wise causal attention
        3. Hidden states = latent representations
        4. Prediction head predicts next scale (auxiliary task)

    Token sequence: [CLS(1), S1(4), S2(16), S3(64), S4(256)] = 341 tokens

    Attention pattern:
        - CLS: Sees ALL tokens (global aggregation)
        - S1: CLS + S1 (bidirectional within S1)
        - S2: CLS + S1 + S2 (bidirectional within S2)
        - S3: CLS + S1 + S2 + S3 (bidirectional within S3)
        - S4: All tokens (bidirectional within S4)
    """

    def __init__(self, config: Optional[VAREncoderConfig] = None, **kwargs):
        super().__init__()

        # Handle config
        if config is not None:
            self.config = config
        else:
            self.config = VAREncoderConfig(**kwargs)

        cfg = self.config
        use_rope = (cfg.pos_type == "rope")

        # Multi-scale patch embedding (real patches, not mask tokens!)
        self.patch_embed = MultiScalePatchEmbed(
            dim=cfg.dim,
            patch_size=cfg.patch_size,
            pos_type=cfg.pos_type,
        )

        # Transformer encoder
        self.transformer = TransformerEncoder(
            dim=cfg.dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            drop=cfg.drop,
            attn_drop=cfg.attn_drop,
            drop_path=cfg.drop_path,
            use_flash_attn=cfg.use_flash_attn,
            use_rope=use_rope,
        )

        # Prediction head: 1→4 (shared across all scale transitions)
        self.pred_head = PredictionHead(cfg.dim)

        # Scale boundaries and parent-child mappings
        self.boundaries = get_scale_boundaries()  # [0, 1, 5, 21, 85, 341]
        self.mappings = get_parent_child_mapping()  # [(0,1,1,5), (1,5,5,21), ...]

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        self.apply(self._init_weights_fn)

    def _init_weights_fn(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        images: torch.Tensor,
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            images: [B, 3, 256, 256] input images
            return_loss: If True, compute and return loss

        Returns:
            Dictionary containing:
                - 'loss': scalar loss (if return_loss=True)
                - 'hidden': [B, 341, D] hidden states (representations!)
                - 'cls': [B, D] CLS token representation
        """
        # 1. Embed real patches: tokens [B, 341, D], targets [B, 340, D]
        # tokens = CLS + patches with position encoding
        # targets = raw patch embeddings WITHOUT position (for loss computation)
        tokens, targets = self.patch_embed(images)

        # Get coords for RoPE if needed
        coords = self.patch_embed.get_coords() if self.config.pos_type == "rope" else None

        # 2. Transformer forward: [B, 341, D]
        # hidden states ARE the latent representations
        hidden = self.transformer(tokens, coords=coords)

        output = {
            'hidden': hidden,
            'cls': hidden[:, 0],  # CLS token for downstream tasks
        }

        # 3. Compute prediction loss (auxiliary task)
        if return_loss:
            loss = self.compute_loss(hidden, targets)
            output['loss'] = loss

        return output

    def compute_loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute next-scale prediction loss.

        Each hidden state predicts 4 tokens in the NEXT scale.
        This is an auxiliary task that forces hidden to be a good representation.

        Args:
            hidden: [B, 341, D] transformer hidden states (includes CLS)
            targets: [B, 340, D] raw patch embeddings (detached, no position, no CLS)

        Returns:
            loss: scalar tensor

        Prediction mapping (CLS excluded since it sees all):
            h_S1  [1:5]   → predict S2 targets [4:20]    (4→16)
            h_S2  [5:21]  → predict S3 targets [20:84]   (16→64)
            h_S3  [21:85] → predict S4 targets [84:340]  (64→256)
        """
        B = hidden.shape[0]
        total_loss = 0.0

        # Mappings: [(parent_start, parent_end, child_start, child_end), ...]
        # parent indices in hidden (includes CLS at 0)
        # child indices in targets (no CLS, starts at 0)
        for p_start, p_end, c_start, c_end in self.mappings:
            # Get parent hidden states
            parent_hidden = hidden[:, p_start:p_end]  # [B, n_parents, D]

            # Predict children: [B, n_parents, D] → [B, n_parents*4, D]
            pred = self.pred_head(parent_hidden)

            # Get target children
            # In targets: S1=[0:4], S2=[4:20], S3=[20:84], S4=[84:340]
            # Child indices in hidden: c_start:c_end
            # Child indices in targets: c_start-1:c_end-1 (offset by 1 since no CLS)
            target_children = targets[:, c_start - 1:c_end - 1]  # [B, n_children, D]

            # Compute loss
            if self.config.loss_type == "mse":
                scale_loss = F.mse_loss(pred, target_children)
            elif self.config.loss_type == "cosine":
                pred_norm = F.normalize(pred, dim=-1)
                target_norm = F.normalize(target_children, dim=-1)
                cos_sim = (pred_norm * target_norm).sum(dim=-1)
                scale_loss = (1 - cos_sim).mean()
            else:
                raise ValueError(f"Unknown loss type: {self.config.loss_type}")

            total_loss += scale_loss

        return total_loss / len(self.mappings)

    def encode(
        self,
        images: torch.Tensor,
        output_type: Literal["cls", "last_scale", "all_mean", "concat", "multi_scale"] = "cls",
    ) -> torch.Tensor:
        """Extract image representations for downstream tasks.

        NOTE: Prediction head is NOT used here - we use hidden states directly!

        Args:
            images: [B, 3, 256, 256] input images
            output_type:
                - "cls": Return CLS token [B, D]
                - "last_scale": Return mean of S4 tokens [B, D]
                - "all_mean": Return mean of all patch tokens [B, D]
                - "concat": Return [CLS; mean(S4)] [B, 2D]
                - "multi_scale": Return [CLS; S1_avg; S2_avg; S3_avg; S4_avg] [B, 5D]

        Returns:
            representation: [B, D], [B, 2D], or [B, 5D] depending on output_type
        """
        with torch.no_grad():
            output = self.forward(images, return_loss=False)
            hidden = output['hidden']  # [B, 341, D]

        if output_type == "cls":
            return hidden[:, 0]

        elif output_type == "last_scale":
            # S4 tokens: indices 85:341
            s4_start = self.boundaries[-2]  # 85
            s4_end = self.boundaries[-1]    # 341
            return hidden[:, s4_start:s4_end].mean(dim=1)

        elif output_type == "all_mean":
            # Mean of all patch tokens (excluding CLS)
            return hidden[:, 1:].mean(dim=1)

        elif output_type == "concat":
            cls_token = hidden[:, 0]
            s4_start = self.boundaries[-2]
            s4_end = self.boundaries[-1]
            s4_mean = hidden[:, s4_start:s4_end].mean(dim=1)
            return torch.cat([cls_token, s4_mean], dim=-1)

        elif output_type == "multi_scale":
            # [CLS, S1_avg, S2_avg, S3_avg, S4_avg]
            cls_token = hidden[:, 0]
            scale_means = []
            for i in range(NUM_SCALES):
                start = self.boundaries[i + 1]
                end = self.boundaries[i + 2]
                scale_means.append(hidden[:, start:end].mean(dim=1))
            return torch.cat([cls_token] + scale_means, dim=-1)

        else:
            raise ValueError(f"Unknown output_type: {output_type}")

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= sum(p.numel() for p in self.patch_embed.parameters())
        return n_params


def vare_base(**kwargs) -> VAREncoder:
    """VAR-Encoder Base (ViT-B/16 scale): ~86M params."""
    config = VAREncoderConfig(
        dim=768,
        depth=12,
        num_heads=12,
        **kwargs,
    )
    return VAREncoder(config)


def vare_large(**kwargs) -> VAREncoder:
    """VAR-Encoder Large (ViT-L/16 scale): ~307M params."""
    config = VAREncoderConfig(
        dim=1024,
        depth=24,
        num_heads=16,
        **kwargs,
    )
    return VAREncoder(config)


def vare_huge(**kwargs) -> VAREncoder:
    """VAR-Encoder Huge (ViT-H/16 scale): ~632M params."""
    config = VAREncoderConfig(
        dim=1280,
        depth=32,
        num_heads=16,
        **kwargs,
    )
    return VAREncoder(config)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing VAR-Encoder")
    print("=" * 60)

    # Test with RoPE + Cosine (recommended config)
    print("\n--- Testing with RoPE + Cosine Loss ---")
    model = vare_base(loss_type="cosine", pos_type="rope")

    images = torch.randn(2, 3, 256, 256)
    output = model(images, return_loss=True)

    print(f"Loss: {output['loss'].item():.4f}")
    print(f"Hidden shape: {output['hidden'].shape}")  # [2, 341, 768]
    print(f"CLS shape: {output['cls'].shape}")  # [2, 768]

    # Test encoding (prediction head NOT used!)
    print("\nRepresentation shapes (pred_head NOT used):")
    for out_type in ["cls", "last_scale", "all_mean", "concat", "multi_scale"]:
        rep = model.encode(images, output_type=out_type)
        print(f"  {out_type}: {rep.shape}")

    # Test with MLP position
    print("\n--- Testing with MLP Position Encoding ---")
    model_mlp = vare_base(loss_type="cosine", pos_type="mlp")
    output_mlp = model_mlp(images, return_loss=True)
    print(f"MLP Loss: {output_mlp['loss'].item():.4f}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")

    # Verify 1→4 mapping (CLS→S1 removed)
    print("\n--- Verifying 1→4 Prediction Mapping ---")
    print("(CLS→S1 removed since CLS sees all scales)")
    mappings = get_parent_child_mapping()
    for p_start, p_end, c_start, c_end in mappings:
        n_parents = p_end - p_start
        n_children = c_end - c_start
        print(f"  h[{p_start}:{p_end}] ({n_parents}) → predict targets[{c_start-1}:{c_end-1}] ({n_children})")
        assert n_children == n_parents * 4, "1→4 mapping violated!"
    print(f"Total mappings: {len(mappings)} (S1→S2, S2→S3, S3→S4)")
