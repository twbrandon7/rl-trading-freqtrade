from typing import Any, Dict, Optional, Union

import gymnasium as gym
import torch
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.RL.BaseReinforcementLearningModel import (
    BaseReinforcementLearningModel,
)
from pandas import DataFrame
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from user_data.rl.agent import Agent
from user_data.rl.env.my_rl_env import MyRLEnv as UserMyRLEnv


class CleanRlLearner(BaseReinforcementLearningModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(config=kwargs["config"])
        self.max_threads = min(
            self.freqai_info["rl_config"].get("cpu_count", 1),
            max(int(self.max_system_threads / 2), 1),
        )
        torch.set_num_threads(self.max_threads)
        self.reward_params = self.freqai_info["rl_config"]["model_reward_parameters"]
        self.train_env: Union[VecMonitor, SubprocVecEnv, gym.Env] = gym.Env()
        self.eval_env: Union[VecMonitor, SubprocVecEnv, gym.Env] = gym.Env()
        self.eval_callback: Optional[MaskableEvalCallback] = None
        self.model_type = self.freqai_info["rl_config"]["model_type"]
        self.rl_config = self.freqai_info["rl_config"]
        self.df_raw: DataFrame = DataFrame()
        self.continual_learning = self.freqai_info.get("continual_learning", False)

        self.unset_outlier_removal()
        self.dd.model_type = "pytorch"

    def fit(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen, **kwargs):
        env = self.train_env
        next_obs, _ = env.reset(seed=42)
        max_steps = self.config["freqai"]["rl_config"]["max_trade_duration_candles"]
        for step in range(0, max_steps - 1):
            next_obs, reward, terminations, truncations, infos = env.step(0)
            pass
        return Agent()

    # def define_data_pipeline(self, threads=-1) -> Pipeline:
    #     pipe_steps = [
    #         # ("const", ds.VarianceThreshold(threshold=0)),
    #         # ("scaler", SKLearnWrapper(MinMaxScaler(feature_range=(-1, 1)))),
    #     ]
    #     return Pipeline(pipe_steps)

    class MyRLEnv(UserMyRLEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
