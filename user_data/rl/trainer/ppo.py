import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import wandb
from torch.utils.tensorboard import SummaryWriter

from user_data.rl.models.base_model import BaseActorCriticModel
from user_data.rl.trainer.args import Args
from user_data.rl.trainer.network_optimization import NetworkOptimization
from user_data.rl.trainer.trajectory_collector import TrajectoryCollector
from user_data.rl.trainer.vector_env import EnvProvider, SyncVectorEnvFactory


class PpoTrainer:
    def __init__(
        self,
        env_provider: EnvProvider,
        agent: BaseActorCriticModel,
        args: Args,
    ) -> None:
        self.env_provider = env_provider
        self.agent = agent
        self.args = args
        self.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

        self.args.batch_size = int(args.num_envs * args.num_steps)
        self.args.minibatch_size = int(args.batch_size // args.num_minibatches)
        self.args.num_iterations = args.total_timesteps // args.batch_size

    def _init_logger(self) -> SummaryWriter:
        if self.args.track:
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                sync_tensorboard=True,
                config=vars(self.args),
                name=self.run_name,
                monitor_gym=True,
                save_code=True,
            )
        writer = SummaryWriter(
            os.path.join(self.args.tensorboard_log_dir, self.run_name)
        )
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % (
                "\n".join(
                    [f"|{key}|{value}|" for key, value in vars(self.args).items()]
                )
            ),
        )
        return writer

    def _set_random_seed(self):
        # TRY NOT TO MODIFY: seeding
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

    def train(self):
        writer = self._init_logger()
        self._set_random_seed()

        device = torch.device(
            "cuda" if torch.cuda.is_available() and self.args.cuda else "cpu"
        )

        # env setup
        env_factory = SyncVectorEnvFactory(self.env_provider, self.run_name)
        envs = env_factory.make(
            self.args.num_envs, self.args.capture_video, self.args.video_path
        )
        assert isinstance(
            envs.single_action_space, gym.spaces.Discrete
        ), "only discrete action space is supported"

        agent = self.agent
        agent.to(device)
        optimizer = optim.Adam(agent.parameters(), lr=self.args.learning_rate, eps=1e-5)
        network_optimization = NetworkOptimization(self.args, optimizer)

        trajectory_collector = TrajectoryCollector(
            self.args, device, envs, agent, writer
        )

        for iteration in range(1, self.args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if self.args.anneal_lr:
                network_optimization.learning_rate_decay(iteration)

            # collect trajectories
            start_time = time.time()
            b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = (
                trajectory_collector.collect()
            )
            global_step = trajectory_collector.global_step

            # optimize policy
            network_optimization.train(
                global_step=global_step,
                b_obs=b_obs,
                b_logprobs=b_logprobs,
                b_actions=b_actions,
                b_advantages=b_advantages,
                b_returns=b_returns,
                b_values=b_values,
            )

            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar(
                "charts/SPS", int(global_step / (time.time() - start_time)), global_step
            )
