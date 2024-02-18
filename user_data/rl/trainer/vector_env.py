import os
from typing import Protocol

import gymnasium as gym


class EnvProvider(Protocol):
    def __call__(self, render_mode: str) -> gym.Env: ...


def make_env(
    env_provider: EnvProvider,
    idx: int,
    capture_video: bool,
    run_name: str,
    video_path: str = "./videos",
):
    def thunk():
        if capture_video and idx == 0:
            env = env_provider(render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, os.path.join(video_path, run_name))
        else:
            env = env_provider(render_mode=None)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


class SyncVectorEnvFactory:
    def __init__(self, env_provider: EnvProvider, run_name: str) -> None:
        self._env_provider = env_provider
        self._run_name = run_name

    def make(
        self, num_envs: int, capture_video: bool = False, video_path: str = "./videos"
    ):
        return gym.vector.SyncVectorEnv(
            [
                make_env(
                    self._env_provider, i, capture_video, self._run_name, video_path
                )
                for i in range(num_envs)
            ],
        )
