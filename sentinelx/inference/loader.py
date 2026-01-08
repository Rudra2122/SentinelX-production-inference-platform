from typing import Any, Tuple

import torch
import torch.nn as nn

from sentinelx.core.config import ModelConfig
from sentinelx.observability.logging import logger


# ============================================================
# Demo Models
# ============================================================

class DemoClassifier(nn.Module):
    """
    Simple demo classifier.

    Input:
      - input_dim features (from ModelConfig)

    Output:
      - num_classes logits (default = 3)
    """
    def __init__(self, input_dim: int, num_classes: int = 3):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DemoRegressor(nn.Module):
    """
    Simple demo regressor.

    Input:
      - input_dim features (from ModelConfig)

    Output:
      - single regression value
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ============================================================
# Model factory
# ============================================================

def build_model(model_cfg: ModelConfig) -> nn.Module:
    """
    Demo model factory.

    Chooses model type based on model name and uses
    input_dim from ModelConfig.
    """
    input_dim = model_cfg.input_dim

    if "regressor" in model_cfg.name.lower():
        return DemoRegressor(input_dim=input_dim)

    return DemoClassifier(input_dim=input_dim)


# ============================================================
# Loader
# ============================================================

def load_model(model_cfg: ModelConfig) -> Tuple[Any, torch.device]:
    """
    Build and load model onto the configured device.

    Returns:
      - model (nn.Module)
      - device (torch.device)
    """
    logger.info(
        f"[Loader] loading model={model_cfg.name} "
        f"v={model_cfg.version} device={model_cfg.device} "
        f"input_dim={model_cfg.input_dim}"
    )

    model = build_model(model_cfg)
    device = torch.device(model_cfg.device)

    model.to(device)
    model.eval()

    return model, device
