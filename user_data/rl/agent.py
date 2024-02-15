from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch


class Agent:
    def __init__(self) -> None:
        self._model_path = None
        self.model = None

    def __setstate__(self, d):
        self.__dict__ = d
        self.model = torch.load(self._model_path)

    def save(self, path: str):
        # save pytorch model
        model_path = f"{path}.model.pt"
        torch.save(self.model, model_path)
        self._model_path = model_path
        # save agent
        torch.save(
            {
                "pytrainer": self,
            },
            path,
        )

    def predict(
        self, observations: pd.DataFrame, deterministic=True
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        return 0, None
