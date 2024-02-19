from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from user_data.rl.models.base_model import BaseActorCriticModel


class BaseAgent(ABC):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._model_path = None
        self.model = model

    def save(self, path: str):
        # save pytorch model
        model_path = f"{path}.model.pt"
        torch.save(self.model, model_path)
        self._model_path = model_path
        # save agent
        torch.save(
            {
                "pytrainer": self.__class__,
                "agent": self,
            },
            path,
        )

    @staticmethod
    def load_from_checkpoint(pickle_dict: dict):
        agent: "BaseAgent" = pickle_dict["agent"]
        agent.model = torch.load(agent._model_path)
        return agent

    @abstractmethod
    def predict(
        self, observations: pd.DataFrame, deterministic=True
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        pass


class ActorCriticAgent(BaseAgent):
    def __init__(self, model: BaseActorCriticModel) -> None:
        super().__init__(model)

    def predict(
        self, observations: pd.DataFrame, deterministic=True
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        x = torch.from_numpy(observations.values).float()
        model: BaseActorCriticModel = self.model
        action, (log_prob, entropy, value) = model.get_action_and_value(x)
        return action.numpy(), (log_prob.numpy(), entropy.numpy(), value.numpy())
