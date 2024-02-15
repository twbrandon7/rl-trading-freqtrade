from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from datasieve.pipeline import Pipeline
from datasieve.transforms import SKLearnWrapper
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.RL.Base3ActionRLEnv import Actions, Base3ActionRLEnv, Positions
from freqtrade.freqai.RL.BaseReinforcementLearningModel import (
    BaseReinforcementLearningModel,
)
from pandas import DataFrame
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from torch import nn

from user_data.rl.agent import Agent


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

    class MyRLEnv(Base3ActionRLEnv):
        def _get_observation(self):
            """
            This may or may not be independent of action types, user can inherit
            this in their custom "MyRLEnv"
            """
            features_window = self.signal_features[
                (self._current_tick - self.window_size) : self._current_tick
            ]
            if self.add_state_info:
                features_and_state = DataFrame(
                    np.zeros((len(features_window), 3)),
                    columns=["current_profit_pct", "position", "trade_duration"],
                    index=features_window.index,
                )

                features_and_state["current_profit_pct"] = self.get_unrealized_profit()
                features_and_state["position"] = self._position.value
                features_and_state["trade_duration"] = self.get_trade_duration()
                features_and_state = pd.concat(
                    [features_window, features_and_state], axis=1
                )
                return features_and_state
            else:
                return features_window

        def calculate_reward(self, action: int) -> float:

            # first, penalize if the action is not valid
            if not self._is_valid(action):
                return -2

            pnl = self.get_unrealized_profit()
            rew = np.sign(pnl) * (pnl + 1)
            factor = 100.0

            # reward agent for entering trades
            if (
                action in (Actions.Buy.value, Actions.Sell.value)
                and self._position == Positions.Neutral
            ):
                return 25
            # discourage agent from not entering trades
            if action == Actions.Neutral.value and self._position == Positions.Neutral:
                return -1

            max_trade_duration = self.rl_config.get("max_trade_duration_candles", 300)
            trade_duration = self._current_tick - self._last_trade_tick  # type: ignore

            if trade_duration <= max_trade_duration:
                factor *= 1.5
            elif trade_duration > max_trade_duration:
                factor *= 0.5

            # discourage sitting in position
            if self._position in (Positions.Short, Positions.Long) and (
                action == Actions.Neutral.value
                or (action == Actions.Sell.value and self._position == Positions.Short)
                or (action == Actions.Buy.value and self._position == Positions.Long)
            ):
                return -1 * trade_duration / max_trade_duration

            # close position
            if (action == Actions.Buy.value and self._position == Positions.Short) or (
                action == Actions.Sell.value and self._position == Positions.Long
            ):
                if pnl > self.profit_aim * self.rr:
                    factor *= self.rl_config["model_reward_parameters"].get(
                        "win_reward_factor", 2
                    )
                return float(rew * factor)

            return 0.0
