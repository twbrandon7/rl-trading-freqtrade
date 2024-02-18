from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import nn


class BaseActorCriticModel(nn.Module, ABC):
    @abstractmethod
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass
