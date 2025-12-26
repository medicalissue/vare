"""Logging utilities for VAR-Encoder training."""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_logging(
    output_dir: Optional[str] = None,
    log_level: int = logging.INFO,
) -> logging.Logger:
    """Setup logging configuration.

    Args:
        output_dir: Directory to save log files
        log_level: Logging level

    Returns:
        Logger instance
    """
    logger = logging.getLogger("vare")
    logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (if output_dir provided)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = output_dir / f"train_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)

    return logger


def init_wandb(
    project: str = "vare",
    entity: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    run_name: Optional[str] = None,
    tags: Optional[list] = None,
    mode: str = "online",
) -> Optional[Any]:
    """Initialize Weights & Biases logging.

    Args:
        project: W&B project name
        entity: W&B entity (team/user)
        config: Configuration dict to log
        run_name: Name for this run
        tags: Tags for this run
        mode: "online", "offline", or "disabled"

    Returns:
        W&B run object or None if not available
    """
    if not WANDB_AVAILABLE:
        print("wandb not installed, skipping wandb initialization")
        return None

    if mode == "disabled":
        wandb.init(mode="disabled")
        return None

    # Generate run name if not provided
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"vare_{timestamp}"

    run = wandb.init(
        project=project,
        entity=entity,
        config=config,
        name=run_name,
        tags=tags,
        mode=mode,
    )

    return run


def log_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    use_wandb: bool = True,
):
    """Log metrics to wandb and/or console.

    Args:
        metrics: Dictionary of metric names and values
        step: Global step (optional)
        use_wandb: Whether to log to wandb
    """
    if use_wandb and WANDB_AVAILABLE:
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)


def log_model_summary(model, input_shape=(1, 3, 256, 256)):
    """Log model summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 50)
    print("Model Summary")
    print("=" * 50)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    print("=" * 50 + "\n")

    if WANDB_AVAILABLE:
        wandb.summary["total_params"] = total_params
        wandb.summary["trainable_params"] = trainable_params


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricTracker:
    """Track multiple metrics."""

    def __init__(self, *metric_names):
        self.metrics = {name: AverageMeter() for name in metric_names}

    def reset(self):
        for meter in self.metrics.values():
            meter.reset()

    def update(self, **kwargs):
        for name, value in kwargs.items():
            if name in self.metrics:
                self.metrics[name].update(value)

    def get_averages(self) -> Dict[str, float]:
        return {name: meter.avg for name, meter in self.metrics.items()}


if __name__ == "__main__":
    # Test logging
    logger = setup_logging(output_dir="./logs")
    logger.info("Test log message")

    # Test metric tracking
    tracker = MetricTracker("loss", "accuracy")
    for i in range(10):
        tracker.update(loss=0.5 - i * 0.05, accuracy=0.5 + i * 0.05)

    print(tracker.get_averages())
