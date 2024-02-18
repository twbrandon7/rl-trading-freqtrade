import os
import tempfile
from unittest import TestCase
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from user_data.rl.trainer.vector_env import SyncVectorEnvFactory, make_env


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


class TestSyncVectorEnvFactory(TestCase):
    def test_stacked_env(self):
        single_env = gym.make("CartPole-v1", render_mode=None)
        factory = SyncVectorEnvFactory(
            lambda render_mode: gym.make("CartPole-v1", render_mode=render_mode),
            "test_run",
        )
        nb_envs = 2
        envs = factory.make(nb_envs)

        observation, info = envs.reset()

        terminated = [False for _ in range(nb_envs)]
        truncated = [False for _ in range(nb_envs)]
        while not all(terminated) and not all(truncated):
            [observation, reward, terminated, truncated, info] = envs.step(
                envs.action_space.sample()
            )

        # check observation shape
        self.assertEqual(
            observation.shape, (nb_envs, *single_env.observation_space.shape)
        )

        # check reward, terminated, and truncated shape
        self.assertEqual(reward.shape, (nb_envs,))
        self.assertEqual(len(terminated), nb_envs)
        self.assertEqual(len(truncated), nb_envs)

        # check info keys
        self.assertTrue("final_observation" in info)
        self.assertTrue("final_info" in info)

        # check final_observation shape
        self.assertEqual(2, len(info["final_observation"]))
        self.assertTrue(
            all(
                info["final_observation"][i].shape == single_env.observation_space.shape
                for i in range(nb_envs)
            )
        )

        # check final_info keys
        self.assertTrue(all("episode" in i for i in info["final_info"]))

    def test_stacked_env_with_video(self):
        with patch("gymnasium.wrappers.RecordVideo") as mock_record_video:
            # make sure RecordVideo is only called once
            mock_record_video.return_value = gym.make(
                "CartPole-v1", render_mode="rgb_array"
            )

            factory = SyncVectorEnvFactory(
                lambda render_mode: gym.make("CartPole-v1", render_mode=render_mode),
                "test_run",
            )
            nb_envs = 3
            envs = factory.make(nb_envs, capture_video=True, video_path="./videos")

            _, _ = envs.reset()
            terminated = [False for _ in range(nb_envs)]
            truncated = [False for _ in range(nb_envs)]
            while not all(terminated) and not all(truncated):
                [_, _, terminated, truncated, _] = envs.step(envs.action_space.sample())

            # make sure RecordVideo is only called once
            self.assertEqual(mock_record_video.call_count, 1)
