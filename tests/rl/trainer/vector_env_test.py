import os
import tempfile
from unittest import TestCase

import gymnasium as gym
import numpy as np

from user_data.rl.trainer.vector_env import make_env


class TestMakeEnv(TestCase):
    def test_make_env_without_video(self):
        env_provider = make_env(
            lambda render_mode: gym.make("CartPole-v1", render_mode=render_mode),
            0,
            False,
            "test_run",
        )
        env = env_provider()
        # Check if the environment is a gym environment
        self.assertIsInstance(env, gym.Env)

        observation, info = env.reset()
        # Check if the observation is a numpy array and info is a dictionary
        self.assertIsInstance(observation, np.ndarray)
        self.assertIsInstance(info, dict)

        terminated, truncated = False, False
        while not terminated and not truncated:
            [observation, reward, terminated, truncated, info] = env.step(
                env.action_space.sample()
            )

        # Check if RecordEpisodeStatistics is applied
        self.assertTrue("episode" in info)

    def test_make_env_with_video(self):
        with tempfile.TemporaryDirectory() as tempdir:
            env_provider = make_env(
                lambda render_mode: gym.make("CartPole-v1", render_mode=render_mode),
                0,
                True,
                "test_run",
                tempdir,
            )
            env = env_provider()
            # Check if the environment is a gym environment
            self.assertIsInstance(env, gym.Env)

            observation, info = env.reset()
            # Check if the observation is a numpy array and info is a dictionary
            self.assertIsInstance(observation, np.ndarray)
            self.assertIsInstance(info, dict)

            terminated, truncated = False, False
            while not terminated and not truncated:
                [observation, reward, terminated, truncated, info] = env.step(
                    env.action_space.sample()
                )

            # Check if RecordEpisodeStatistics is applied
            self.assertTrue("episode" in info)

            # Check if the video file is created
            self.assertTrue(
                os.path.exists(
                    os.path.join(tempdir, "test_run", "rl-video-episode-0.mp4")
                )
            )
