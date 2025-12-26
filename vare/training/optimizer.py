"""Optimizer and scheduler utilities for VAR-Encoder training."""

import math
from typing import Tuple, Optional, List

import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(
    model: torch.nn.Module,
    lr: float = 1.5e-4,
    weight_decay: float = 0.05,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    filter_bias_and_bn: bool = True,
) -> optim.AdamW:
    """Build AdamW optimizer with weight decay filtering.

    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay coefficient
        betas: Adam betas
        eps: Adam epsilon
        filter_bias_and_bn: If True, don't apply weight decay to bias and norm params

    Returns:
        AdamW optimizer
    """
    if filter_bias_and_bn:
        # Separate parameters into decay and no_decay groups
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # No weight decay for biases and LayerNorm weights
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
    else:
        param_groups = model.parameters()

    optimizer = optim.AdamW(
        param_groups,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay if not filter_bias_and_bn else 0.0,
    )

    return optimizer


def build_scheduler(
    optimizer: optim.Optimizer,
    num_epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 40,
    min_lr: float = 1e-6,
    warmup_start_lr: float = 1e-6,
) -> LambdaLR:
    """Build cosine annealing scheduler with linear warmup.

    Args:
        optimizer: Optimizer to schedule
        num_epochs: Total number of training epochs
        steps_per_epoch: Number of steps per epoch
        warmup_epochs: Number of warmup epochs
        min_lr: Minimum learning rate
        warmup_start_lr: Initial learning rate for warmup

    Returns:
        LambdaLR scheduler
    """
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    # Get base learning rate
    base_lr = optimizer.param_groups[0]["lr"]

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            alpha = current_step / warmup_steps
            return (warmup_start_lr + alpha * (base_lr - warmup_start_lr)) / base_lr
        else:
            # Cosine annealing
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return (min_lr + cosine_decay * (base_lr - min_lr)) / base_lr

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler


class CosineWarmupScheduler:
    """Cosine annealing scheduler with warmup, epoch-based."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        num_epochs: int,
        warmup_epochs: int = 40,
        min_lr: float = 1e-6,
        warmup_start_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.current_epoch = 0

    def step(self, epoch: Optional[int] = None):
        """Update learning rate for the given epoch."""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        lr = self._compute_lr(self.current_epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _compute_lr(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            # Linear warmup
            alpha = epoch / self.warmup_epochs
            return self.warmup_start_lr + alpha * (self.base_lr - self.warmup_start_lr)
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.num_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + cosine_decay * (self.base_lr - self.min_lr)

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


class GradScaler:
    """Wrapper for mixed precision gradient scaling."""

    def __init__(
        self,
        enabled: bool = True,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        self.enabled = enabled
        if enabled:
            self.scaler = torch.amp.GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
            )
        else:
            self.scaler = None

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: optim.Optimizer):
        if self.enabled:
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self):
        if self.enabled:
            self.scaler.update()

    def unscale_(self, optimizer: optim.Optimizer):
        if self.enabled:
            self.scaler.unscale_(optimizer)


def get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    """Compute gradient norm for logging."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]

    if len(parameters) == 0:
        return torch.tensor(0.0)

    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([
            torch.norm(p.grad.detach(), norm_type).to(device)
            for p in parameters
        ]),
        norm_type,
    )

    return total_norm


if __name__ == "__main__":
    # Test optimizer and scheduler
    model = torch.nn.Linear(10, 10)
    optimizer = build_optimizer(model, lr=1e-3)
    print(f"Optimizer: {optimizer}")

    # Test epoch-based scheduler
    scheduler = CosineWarmupScheduler(
        optimizer,
        num_epochs=100,
        warmup_epochs=10,
    )

    lrs = []
    for epoch in range(100):
        scheduler.step(epoch)
        lrs.append(scheduler.get_lr())

    print(f"LR at epoch 0: {lrs[0]:.6f}")
    print(f"LR at epoch 10: {lrs[10]:.6f}")
    print(f"LR at epoch 50: {lrs[50]:.6f}")
    print(f"LR at epoch 99: {lrs[99]:.6f}")
