import os
import tempfile
import unittest
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from user_data.rl.agent import BaseAgent


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)


class DummyAgent(BaseAgent):
    def predict(
        self, observations: pd.DataFrame, deterministic=True
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        return self.model(observations), None


class AgentTest(unittest.TestCase):
    def test_save(self):
        """Test saving agent and model to disk.
        """
        model = DummyAgent(DummyModel())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.zip")
            model.save(path)
            self.assertTrue(model._model_path is not None)
            # check file exists
            self.assertTrue(os.path.exists(model._model_path))
            self.assertTrue(os.path.exists(path))

    def test_load_from_checkpoint(self):
        """Test loading agent and model from disk.
        This test also checks if the model is correctly loaded by making a prediction.
        """
        agent = DummyAgent(DummyModel())
        # set weights to 2
        agent.model.fc.weight.data.fill_(2)
        # set bias to 0
        agent.model.fc.bias.data.fill_(0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.zip")
            agent.save(path)
            agent = BaseAgent.load_from_checkpoint(torch.load(path))
            self.assertTrue(agent._model_path is not None)
            prediction, _ = agent.predict(torch.tensor([1.0]))
            self.assertEqual(prediction, torch.tensor([2.0]))


if __name__ == "__main__":
    unittest.main()
