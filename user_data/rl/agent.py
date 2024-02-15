from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch


class Agent:
    def __init__(self) -> None:
        self._model_path = None
        self.model = None

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
        agent: "Agent" = pickle_dict["agent"]
        agent.model = torch.load(agent._model_path)
        return agent

    def predict(
        self, observations: pd.DataFrame, deterministic=True
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        return 0, None
