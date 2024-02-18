import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from user_data.rl.trainer.args import Args


class NetworkOptimization:
    def __init__(
        self, args: Args, optimizer: torch.optim.Optimizer, writer: SummaryWriter
    ):
        self.args = args
        self.optimizer = optimizer
        self.writer = writer

    def learning_rate_decay(self, iteration: int):
        frac = 1.0 - (iteration - 1.0) / self.args.num_iterations
        lrnow = frac * self.args.learning_rate
        self.optimizer.param_groups[0]["lr"] = lrnow

    def train(
        self,
        global_step: int,
        b_obs: torch.Tensor,
        b_logprobs: torch.Tensor,
        b_actions: torch.Tensor,
        b_advantages: torch.Tensor,
        b_returns: torch.Tensor,
        b_values: torch.Tensor,
    ):
        # Optimizing the policy and value network
        b_inds = np.arange(self.args.batch_size)
        clipfracs = []
        for epoch in range(self.args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.args.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > self.args.clip_coef)
                        .float()
                        .mean()
                        .item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if self.args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - self.args.ent_coef * entropy_loss
                    + v_loss * self.args.vf_coef
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.args.parameters(), self.args.max_grad_norm
                )
                self.optimizer.step()

            if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        learning_rate = self.optimizer.param_groups[0]["lr"]

        self._log_metrics(
            global_step=global_step,
            learning_rate=learning_rate,
            v_loss=v_loss,
            pg_loss=pg_loss,
            entropy_loss=entropy_loss,
            old_approx_kl=old_approx_kl,
            approx_kl=approx_kl,
            clipfracs=clipfracs,
            explained_var=explained_var,
        )

    def _log_metrics(
        self,
        global_step: int,
        learning_rate: float,
        v_loss: torch.Tensor,
        pg_loss: torch.Tensor,
        entropy_loss: torch.Tensor,
        old_approx_kl: torch.Tensor,
        approx_kl: torch.Tensor,
        clipfracs: list,
        explained_var: float,
    ):
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        self.writer.add_scalar("charts/learning_rate", learning_rate, global_step)
        self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        self.writer.add_scalar(
            "losses/old_approx_kl", old_approx_kl.item(), global_step
        )
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        self.writer.add_scalar("losses/explained_variance", explained_var, global_step)
