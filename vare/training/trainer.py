"""Training loop for VAR-Encoder."""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .optimizer import (
    build_optimizer,
    CosineWarmupScheduler,
    GradScaler,
    get_grad_norm,
)


@dataclass
class TrainerConfig:
    """Configuration for Trainer."""
    # Training
    epochs: int = 300
    lr: float = 1.5e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 40
    min_lr: float = 1e-6
    grad_clip: float = 1.0

    # Mixed precision
    precision: str = "bf16"  # "fp32", "fp16", "bf16"

    # Checkpointing
    save_every: int = 10
    output_dir: str = "./checkpoints"

    # Logging
    log_every: int = 100
    use_wandb: bool = True


class Trainer:
    """Trainer for VAR-Encoder."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        config: Optional[TrainerConfig] = None,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainerConfig()
        self.device = device

        # Setup optimizer
        self.optimizer = build_optimizer(
            model,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # Setup scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            num_epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
            min_lr=self.config.min_lr,
        )

        # Setup mixed precision
        self.use_amp = self.config.precision != "fp32"
        self.amp_dtype = self._get_amp_dtype()
        self.scaler = GradScaler(enabled=(self.config.precision == "fp16"))

        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.global_step = 0

    def _get_amp_dtype(self) -> torch.dtype:
        """Get AMP dtype based on config."""
        if self.config.precision == "bf16":
            return torch.bfloat16
        elif self.config.precision == "fp16":
            return torch.float16
        else:
            return torch.float32

    def train(self):
        """Run full training loop."""
        print(f"Starting training for {self.config.epochs} epochs")
        print(f"  - Device: {self.device}")
        print(f"  - Precision: {self.config.precision}")
        print(f"  - Batch size: {self.train_loader.batch_size}")
        print(f"  - Steps per epoch: {len(self.train_loader)}")
        print(f"  - Total parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch

            # Update learning rate
            self.scheduler.step(epoch)

            # Train one epoch
            train_metrics = self.train_epoch(epoch)

            # Validation
            if self.val_loader is not None:
                val_metrics = self.validate()
                train_metrics.update(val_metrics)

            # Log metrics
            if self.config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "epoch": epoch,
                    "lr": self.scheduler.get_lr(),
                    **train_metrics,
                })

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch)

        # Save final checkpoint
        self.save_checkpoint(self.config.epochs - 1, is_final=True)
        print("Training completed!")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0
        start_time = time.time()

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config.epochs}",
            leave=True,
        )

        for batch_idx, images in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)

            # Forward pass with AMP
            with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                output = self.model(images, return_loss=True)
                loss = output["loss"]

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.scheduler.get_lr():.2e}",
            })

            # Detailed logging
            if self.global_step % self.config.log_every == 0:
                grad_norm = get_grad_norm(self.model.parameters())

                if self.config.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/grad_norm": grad_norm.item(),
                        "train/lr": self.scheduler.get_lr(),
                        "train/step": self.global_step,
                    })

        # Epoch metrics
        epoch_time = time.time() - start_time
        avg_loss = total_loss / num_batches

        print(f"Epoch {epoch} completed in {epoch_time:.1f}s, avg loss: {avg_loss:.4f}")

        return {
            "train/epoch_loss": avg_loss,
            "train/epoch_time": epoch_time,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for images in tqdm(self.val_loader, desc="Validating", leave=False):
            images = images.to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                output = self.model(images, return_loss=True)
                loss = output["loss"]

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        return {"val/loss": avg_loss}

    def save_checkpoint(self, epoch: int, is_final: bool = False):
        """Save model checkpoint."""
        filename = "checkpoint_final.pt" if is_final else f"checkpoint_epoch_{epoch:04d}.pt"
        filepath = self.output_dir / filename

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }

        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint to {filepath}")

        # Also save latest
        latest_path = self.output_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]

        print(f"Loaded checkpoint from {filepath}, resuming from epoch {self.current_epoch}")

    def resume(self, checkpoint_dir: Optional[str] = None):
        """Resume training from latest checkpoint."""
        checkpoint_dir = Path(checkpoint_dir or self.config.output_dir)
        latest_path = checkpoint_dir / "checkpoint_latest.pt"

        if latest_path.exists():
            self.load_checkpoint(str(latest_path))
        else:
            print("No checkpoint found, starting from scratch")


if __name__ == "__main__":
    from ..model import vare_base

    # Create dummy model and data
    model = vare_base()

    # Create dummy dataloader
    dummy_data = torch.randn(64, 3, 256, 256)
    dummy_dataset = torch.utils.data.TensorDataset(dummy_data)
    dummy_loader = DataLoader(dummy_dataset, batch_size=8)

    # Create trainer
    config = TrainerConfig(
        epochs=2,
        log_every=5,
        save_every=1,
        use_wandb=False,
    )

    trainer = Trainer(
        model=model,
        train_loader=dummy_loader,
        config=config,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Run training
    trainer.train()
