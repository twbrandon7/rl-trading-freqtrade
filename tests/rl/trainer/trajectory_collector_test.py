import unittest
from typing import Tuple
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import torch
from pandas import DataFrame
from torch.nn.modules import Module

from user_data.rl.agent import BaseActorCriticAgent
from user_data.rl.trainer.args import Args
from user_data.rl.trainer.trajectory_collector import TrajectoryCollector


class DummyActorCriticAgent(BaseActorCriticAgent):
    def __init__(self, model: Module, args: Args) -> None:
        super().__init__(model)
        self.args = args
        self._steps = 0
        self.dummy_acton = torch.arange(self.args.num_envs)
        self.dummy_logprob = torch.arange(self.args.num_envs) + 1
        self.dummy_entropy = torch.arange(self.args.num_envs) + 2
        self.dummy_critic = torch.arange(self.args.num_envs) + 3

    def predict(
        self, observations: DataFrame, deterministic=True
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, ...] | None]:
        result = (
            self.dummy_acton + self._steps,
            self.dummy_logprob + self._steps,
            self.dummy_entropy + self._steps,
            self.dummy_critic + self._steps,
        )
        self._steps += 1
        return result

    def predict_critic(self, observations: DataFrame) -> np.ndarray:
        return self.dummy_critic + self._steps


class TrajectoryCollectorTest(unittest.TestCase):
    def setUp(self):
        self.args = Args()
        self.args.num_envs = 3
        self.device = torch.device("cpu")
        self.envs = gym.vector.SyncVectorEnv(
            [lambda: gym.make("CartPole-v1") for _ in range(self.args.num_envs)]
        )
        self.dummy_first_obs = np.zeros(
            (self.args.num_envs,) + self.envs.single_observation_space.shape
        )
        self.dummy_first_obs[::] = 3
        self.envs.reset = MagicMock(
            return_value=(
                self.dummy_first_obs,
                None,
            )
        )
        self.agent = DummyActorCriticAgent(torch.nn.Linear(4, 2), self.args)
        self.writer = patch("torch.utils.tensorboard.SummaryWriter").start()

    def tearDown(self):
        self.envs.close()
        self.writer.stop()

    def test_collect_handles_full_steps(self):
        """
        Tests that collect() runs for the full specified num_steps, and logs
        episodic metrics if `final_info` is present in env step returns.
        """
        self.collector = TrajectoryCollector(
            self.args, self.device, self.envs, self.agent, self.writer
        )

        self.envs.step = MagicMock(
            return_value=(
                np.zeros(
                    (self.args.num_envs,) + self.envs.single_observation_space.shape
                ),
                np.zeros(self.args.num_envs),
                np.zeros(self.args.num_envs),
                np.zeros(self.args.num_envs),
                [
                    {"final_info": {"episode": {"r": 123, "l": 456}}}
                    for _ in range(self.args.num_envs)
                ],
            )
        )

        observations, logprobs, actions, advantages, returns, values = (
            self.collector.collect()
        )

        expected_obs = np.concatenate(
            [self.dummy_first_obs]
            + [
                np.zeros_like(self.dummy_first_obs)
                for _ in range(self.args.num_steps - 1)
            ],
            axis=0,
        )
        self.assertTrue(np.array_equal(observations, expected_obs))

        expected_logprobs = np.concatenate(
            [self.agent.dummy_logprob + i for i in range(self.args.num_steps)],
        )
        self.assertTrue(np.array_equal(logprobs, expected_logprobs))

        expected_actions = np.concatenate(
            [self.agent.dummy_acton + i for i in range(self.args.num_steps)],
        )
        self.assertTrue(np.array_equal(actions, expected_actions))

        expected_critics = np.concatenate(
            [self.agent.dummy_critic + i for i in range(self.args.num_steps)],
        )
        self.assertTrue(np.array_equal(values, expected_critics))

        # assert the shape of advantages and returns equal to the shape of critic
        self.assertEqual(advantages.shape, expected_critics.shape)
        self.assertEqual(returns.shape, expected_critics.shape)

    def test_calculate_advantages(self):
        args = Args()
        args.num_steps = 4
        args.num_envs = 3

        self.collector = TrajectoryCollector(
            args, self.device, self.envs, self.agent, self.writer
        )

        # Define test inputs
        next_value = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        rewards = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2],
            ],
            requires_grad=True,
        )
        dones = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            requires_grad=True,
        )
        next_done = torch.tensor([0.0, 0.0, 1.0], requires_grad=True)
        values = torch.tensor(
            [
                [0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7],
                [0.8, 0.9, 1.0],
                [1.1, 1.2, 1.3],
            ],
            requires_grad=True,
        )

        # Call the function
        advantages, returns = self.collector._calculate_advantages(
            next_value=next_value,
            rewards=rewards,
            dones=dones,
            next_done=next_done,
            values=values,
            gae_lambda=0.95,
            gamma=0.99,
        )

        expected_advantages = np.array(
            [
                [2.661036, 3.764306, 2.396804],
                [2.409396, 3.4772, 1.917919],
                [1.826045, 2.85614, 1.09295],
                [0.89, 1.88, -0.1],
            ]
        ).astype(np.float32)

        expected_returns = np.array(
            [
                [2.861036, 4.064307, 2.796804],
                [2.909396, 4.0772, 2.61792],
                [2.626045, 3.75614, 2.09295],
                [1.99, 3.08, 1.2],
            ]
        ).astype(np.float32)

        # Perform assertions
        # Add your assertions based on expected results
        np.testing.assert_array_equal(
            advantages.detach().numpy().round(6).astype(np.float32),
            expected_advantages.round(6),
        )
        np.testing.assert_array_equal(
            returns.detach().numpy().round(6).astype(np.float32),
            expected_returns.round(6),
        )
