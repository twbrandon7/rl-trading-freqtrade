import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from user_data.rl.models.base_model import BaseActorCriticModel
from user_data.rl.trainer.args import Args


class TrajectoryCollector:
    def __init__(
        self,
        args: Args,
        device: torch.device,
        envs: gym.vector.VectorEnv,
        agent: BaseActorCriticModel,
        writer: SummaryWriter,
    ) -> None:
        self.args = args
        self.device = device
        self.envs = envs
        self.agent = agent
        self.writer = writer

        # ALGO Logic: Storage setup
        self.obs = torch.zeros(
            (args.num_steps, args.num_envs) + envs.single_observation_space.shape
        ).to(device)
        self.actions = torch.zeros(
            (args.num_steps, args.num_envs) + envs.single_action_space.shape
        ).to(device)
        self.logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        # TRY NOT TO MODIFY: start the game
        self.global_step = 0
        self.next_obs, _ = envs.reset(seed=args.seed)
        self.next_obs = torch.Tensor(self.next_obs).to(device)
        self.next_done = torch.zeros(args.num_envs).to(device)

        self.global_step = 0

    def collect(self):
        for step in range(0, self.args.num_steps):
            self.global_step += self.args.num_envs
            self.obs[step] = self.next_obs
            self.dones[step] = self.next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(
                    self.next_obs
                )
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            self.next_obs, reward, terminations, truncations, infos = self.envs.step(
                action.cpu().numpy()
            )
            self.next_done = np.logical_or(terminations, truncations)
            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            self.next_obs, self.next_done = torch.Tensor(self.next_obs).to(
                self.device
            ), torch.Tensor(self.next_done).to(self.device)

            if "final_info" in infos:
                self._log_final_infos(infos["final_info"])

        # bootstrap value if not done
        with torch.no_grad():
            next_value = self.agent.get_value(self.next_obs).reshape(1, -1)
            advantages, returns = self._calculate_advantages(
                next_value=next_value,
                rewards=self.rewards,
                dones=self.dones,
                next_done=self.next_done,
                values=self.values,
                gae_lambda=self.args.gae_lambda,
                gamma=self.args.gamma,
            )

        # flatten the batch
        b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values

    def _calculate_advantages(
        self,
        next_value: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_done: torch.Tensor,
        values: torch.Tensor,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
    ):
        advantages = torch.zeros_like(rewards).to(self.device)
        lastgaelam = 0
        for t in reversed(range(self.args.num_steps)):
            if t == self.args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = (
                delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + values
        return advantages, returns

    def _log_final_infos(self, infos):
        for info in infos:
            if not info or "episode" not in info:
                continue
            print(
                "global_step={}, episodic_return={}".format(
                    self.global_step, info["episode"]["r"]
                )
            )
            self.writer.add_scalar(
                "charts/episodic_return", info["episode"]["r"], self.global_step
            )
            self.writer.add_scalar(
                "charts/episodic_length", info["episode"]["l"], self.global_step
            )
