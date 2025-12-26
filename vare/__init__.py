"""VAR-Encoder: Multi-Scale Autoregressive Vision Encoder.

A self-supervised vision encoder that learns image representations
by predicting finer-scale patches from coarser-scale context.

Example:
    >>> from vare import vare_base
    >>> model = vare_base()
    >>> images = torch.randn(2, 3, 256, 256)
    >>> output = model(images, return_loss=True)
    >>> loss = output['loss']
    >>> representation = model.encode(images, output_type='cls')
"""

__version__ = "0.1.0"

from .model import (
    VAREncoder,
    VAREncoderConfig,
    vare_base,
    vare_large,
    vare_huge,
)
from .data import (
    build_dataloader,
    build_eval_dataloader,
    build_train_transform,
    build_val_transform,
)
from .training import (
    Trainer,
    TrainerConfig,
)

__all__ = [
    # Version
    "__version__",
    # Model
    "VAREncoder",
    "VAREncoderConfig",
    "vare_base",
    "vare_large",
    "vare_huge",
    # Data
    "build_dataloader",
    "build_eval_dataloader",
    "build_train_transform",
    "build_val_transform",
    # Training
    "Trainer",
    "TrainerConfig",
]
