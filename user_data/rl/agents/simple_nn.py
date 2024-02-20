from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical

from user_data.rl.agents.base_agent import BaseAgent
from user_data.rl.models.base_model import BaseActorCriticModel
from user_data.rl.models.misc import layer_init


class SimpleNnActorCriticModel(BaseActorCriticModel):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, output_dim), std=0.01),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class ActorCriticAgent(BaseAgent):
    def __init__(self, model: SimpleNnActorCriticModel) -> None:
        super().__init__(model)

    def predict(
        self, observations: pd.DataFrame, deterministic=True
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        x = torch.from_numpy(observations.values).float()
        model: SimpleNnActorCriticModel = self.model
        action, (log_prob, entropy, value) = model.get_action_and_value(x)
        return action.numpy(), (log_prob.numpy(), entropy.numpy(), value.numpy())
