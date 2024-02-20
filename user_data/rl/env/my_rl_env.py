from typing import Any

import numpy as np
import pandas as pd
from freqtrade.freqai.RL.Base3ActionRLEnv import Actions, Base3ActionRLEnv, Positions
from gymnasium.core import ObsType


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
            features_and_state = pd.DataFrame(
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

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        return super().reset(seed)

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
