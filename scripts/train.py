#!/usr/bin/env python
"""Training script for VAR-Encoder.

Usage:
    # Basic training with default config
    python scripts/train.py

    # Train with different model size
    python scripts/train.py model=large

    # Override hyperparameters
    python scripts/train.py train.epochs=100 train.lr=1e-4 data.batch_size=128

    # Use cosine loss instead of MSE
    python scripts/train.py model.loss_type=cosine

    # Disable wandb
    python scripts/train.py wandb.mode=disabled
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from vare.model import VAREncoder, VAREncoderConfig
from vare.data import build_dataloader
from vare.training import Trainer, TrainerConfig
from vare.utils import init_wandb, setup_logging, log_model_summary


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main training function."""
    # Print config
    print("=" * 60)
    print("VAR-Encoder Training")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Set seed
    set_seed(cfg.seed)

    # Setup logging
    logger = setup_logging(output_dir=cfg.train.output_dir)
    logger.info(f"Using device: {cfg.device}")

    # Initialize wandb
    use_wandb = cfg.wandb.mode != "disabled"
    if use_wandb:
        init_wandb(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
        )

    # Build model
    model_config = VAREncoderConfig(
        dim=cfg.model.dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        patch_size=cfg.model.patch_size,
        drop=cfg.model.drop,
        attn_drop=cfg.model.attn_drop,
        drop_path=cfg.model.drop_path,
        loss_type=cfg.model.loss_type,
        use_flash_attn=cfg.model.use_flash_attn,
    )
    model = VAREncoder(model_config)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    # Log model summary
    if use_wandb:
        log_model_summary(model)

    # Build dataloaders
    logger.info(f"Loading data from {cfg.data.root}")
    train_loader = build_dataloader(
        root=cfg.data.root,
        split="train",
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        img_size=cfg.data.img_size,
        pin_memory=cfg.data.pin_memory,
    )
    logger.info(f"Train dataset: {len(train_loader.dataset)} images, {len(train_loader)} batches")

    # Build trainer config
    trainer_config = TrainerConfig(
        epochs=cfg.train.epochs,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        warmup_epochs=cfg.train.warmup_epochs,
        min_lr=cfg.train.min_lr,
        grad_clip=cfg.train.grad_clip,
        precision=cfg.train.precision,
        save_every=cfg.train.save_every,
        output_dir=cfg.train.output_dir,
        log_every=cfg.train.log_every,
        use_wandb=use_wandb,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        config=trainer_config,
        device=cfg.device,
    )

    # Resume from checkpoint if exists
    trainer.resume()

    # Start training
    trainer.train()

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
