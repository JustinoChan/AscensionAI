"""
ppo_model.py — PPOTrainer and GameBuffer classes.

A clean, import-safe module with no side effects (no stdout patching,
no logging to files). Used by train_ppo.py, rollout_worker.py,
train_offline.py, and behavior_clone.py.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F


class GameBuffer:
    """Stores transitions from a single game for PPO training."""

    def __init__(self):
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.action_masks: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []

    def add(self, obs, action, reward, done, mask, log_prob, value):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.action_masks.append(mask)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def __len__(self):
        return len(self.observations)

    def clear(self):
        self.__init__()


class PPOTrainer:
    """Minimal PPO that works with collected transitions, not a gym env."""

    def __init__(self, obs_size: int, n_actions: int, device: str = "cpu",
                 lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_range: float = 0.2, ent_coef: float = 0.001, vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5, n_epochs: int = 4, batch_size: int = 64,
                 net_arch: tuple = (256, 256), target_kl: float = 0.03):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.target_kl = target_kl

        layers = []
        in_dim = obs_size
        for h in net_arch:
            layers.append(torch.nn.Linear(in_dim, h))
            layers.append(torch.nn.Tanh())
            in_dim = h
        self.shared = torch.nn.Sequential(*layers).to(self.device)
        self.policy_head = torch.nn.Linear(in_dim, n_actions).to(self.device)
        self.value_head = torch.nn.Linear(in_dim, 1).to(self.device)

        params = (
            list(self.shared.parameters())
            + list(self.policy_head.parameters())
            + list(self.value_head.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=lr)

        self.total_updates = 0
        self.bc_obs_t = None
        self.bc_actions_t = None
        self.bc_masks_t = None
        self.bc_coef = 0.0
        self.bc_batch_size = batch_size

    def get_lr(self) -> float:
        """Return the current optimizer learning rate."""
        try:
            return float(self.optimizer.param_groups[0].get("lr", 0.0))
        except Exception:
            return 0.0

    def set_lr(self, lr: float) -> None:
        """Reset optimizer learning rate after loading a checkpoint."""
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    def set_bc_reference(self, observations, actions, masks,
                         coef: float = 0.0, batch_size: int | None = None) -> None:
        """Attach behavior-cloning examples used as a small PPO anchor loss."""
        self.bc_coef = float(coef or 0.0)
        self.bc_batch_size = int(batch_size or self.batch_size)
        if self.bc_coef <= 0.0 or observations is None or len(observations) == 0:
            self.bc_obs_t = self.bc_actions_t = self.bc_masks_t = None
            return
        obs = np.array(observations, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        masks = np.array(masks, dtype=np.bool_)
        in_range = (actions >= 0) & (actions < self.n_actions)
        mask_legal = np.zeros_like(in_range, dtype=np.bool_)
        if masks.ndim == 2 and masks.shape[1] == self.n_actions:
            valid_idx = np.where(in_range)[0]
            mask_legal[valid_idx] = masks[valid_idx, actions[valid_idx]]
        finite = np.isfinite(obs).all(axis=1)
        valid = in_range & mask_legal & finite
        if not bool(valid.any()):
            self.bc_obs_t = self.bc_actions_t = self.bc_masks_t = None
            self.bc_coef = 0.0
            return
        obs = obs[valid]
        actions = actions[valid]
        masks = masks[valid]

        self.bc_obs_t = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        )
        self.bc_actions_t = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        )
        self.bc_masks_t = torch.as_tensor(
            masks,
            dtype=torch.bool,
            device=self.device,
        )

    def predict(self, obs: np.ndarray, mask: np.ndarray, deterministic: bool = False):
        """Pick an action, return (action, log_prob, value)."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)

            features = self.shared(obs_t)
            logits = self.policy_head(features)

            logits = logits.masked_fill(~mask_t, -1e8)
            dist = torch.distributions.Categorical(logits=logits)

            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            value = self.value_head(features).squeeze(-1)

        return action.item(), log_prob.item(), value.item()

    def action_probabilities(self, obs: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
        """Return masked action probabilities and value for diagnostics."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            features = self.shared(obs_t)
            logits = self.policy_head(features).masked_fill(~mask_t, -1e8)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            value = self.value_head(features).squeeze(-1)
        return probs.cpu().numpy(), float(value.item())

    def update(self, buffer: GameBuffer) -> dict:
        """Run PPO update on collected transitions. Returns loss stats."""
        if len(buffer) < 2:
            return {}

        obs = np.array(buffer.observations, dtype=np.float32)
        actions = np.array(buffer.actions, dtype=np.int64)
        rewards = np.array(buffer.rewards, dtype=np.float32)
        dones = np.array(buffer.dones, dtype=np.float32)
        masks = np.array(buffer.action_masks, dtype=np.bool_)
        old_log_probs = np.array(buffer.log_probs, dtype=np.float32)
        old_values = np.array(buffer.values, dtype=np.float32)

        in_range = (actions >= 0) & (actions < self.n_actions)
        mask_legal = np.zeros_like(in_range, dtype=np.bool_)
        if masks.ndim == 2 and masks.shape[1] == self.n_actions:
            valid_idx = np.where(in_range)[0]
            mask_legal[valid_idx] = masks[valid_idx, actions[valid_idx]]
        finite = (
            np.isfinite(obs).all(axis=1)
            & np.isfinite(rewards)
            & np.isfinite(old_log_probs)
            & np.isfinite(old_values)
        )
        valid = in_range & mask_legal & finite
        invalid_action_count = int(len(actions) - int(valid.sum()))
        if invalid_action_count:
            obs = obs[valid]
            actions = actions[valid]
            rewards = rewards[valid]
            dones = dones[valid]
            masks = masks[valid]
            old_log_probs = old_log_probs[valid]
            old_values = old_values[valid]
        if len(obs) < 2:
            return {"invalid_action_count": invalid_action_count}

        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = 0.0
            else:
                next_val = old_values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - old_values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + old_values
        raw_advantages = advantages.copy()
        mean_advantage = float(raw_advantages.mean())
        std_advantage = float(raw_advantages.std())
        return_var = float(np.var(returns))
        if return_var > 1e-8:
            explained_variance = float(1.0 - np.var(returns - old_values) / return_var)
        else:
            explained_variance = float("nan")

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.as_tensor(obs, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        old_lp_t = torch.as_tensor(old_log_probs, device=self.device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        masks_t = torch.as_tensor(masks, device=self.device)

        n = len(obs)
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_ent = 0.0
        total_norm_ent = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0
        total_chosen_prob = 0.0
        total_bc_loss = 0.0
        num_batches = 0
        bc_batches = 0
        early_stop = 0

        for _ in range(self.n_epochs):
            epoch_kl = 0.0
            epoch_batches = 0
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = indices[start:end]

                b_obs = obs_t[idx]
                b_act = actions_t[idx]
                b_old_lp = old_lp_t[idx]
                b_adv = adv_t[idx]
                b_ret = ret_t[idx]
                b_mask = masks_t[idx]

                features = self.shared(b_obs)
                logits = self.policy_head(features)
                logits = logits.masked_fill(~b_mask, -1e8)
                dist = torch.distributions.Categorical(logits=logits)

                new_lp = dist.log_prob(b_act)
                entropy_per_sample = dist.entropy()
                entropy = entropy_per_sample.mean()
                # Raw entropy is hard to compare across STS screens because the
                # number of legal actions varies wildly. Normalize by log(N legal
                # actions) so auto-tuning can distinguish "confident" from
                # "forced by the mask" decisions.
                legal_counts = b_mask.sum(dim=1).clamp(min=2).float()
                normalized_entropy = (entropy_per_sample / torch.log(legal_counts)).mean()
                values = self.value_head(features).squeeze(-1)

                ratio = (new_lp - b_old_lp).exp()
                pg_loss1 = -b_adv * ratio
                pg_loss2 = -b_adv * ratio.clamp(1 - self.clip_range, 1 + self.clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                vf_loss = ((values - b_ret) ** 2).mean()

                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy
                bc_loss = None
                if self.bc_coef > 0.0 and self.bc_obs_t is not None:
                    bc_n = self.bc_obs_t.shape[0]
                    k = min(max(1, self.bc_batch_size), bc_n)
                    bc_idx = torch.randint(0, bc_n, (k,), device=self.device)
                    bc_features = self.shared(self.bc_obs_t[bc_idx])
                    bc_logits = self.policy_head(bc_features)
                    bc_logits = bc_logits.masked_fill(~self.bc_masks_t[bc_idx], -1e8)
                    bc_loss = F.cross_entropy(bc_logits, self.bc_actions_t[bc_idx])
                    loss = loss + self.bc_coef * bc_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.shared.parameters())
                    + list(self.policy_head.parameters())
                    + list(self.value_head.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_ent += entropy.item()
                total_norm_ent += normalized_entropy.item()
                with torch.no_grad():
                    batch_kl = (b_old_lp - new_lp).mean().item()
                    total_kl += batch_kl
                    epoch_kl += batch_kl
                    epoch_batches += 1
                    total_clip_frac += ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                    total_chosen_prob += new_lp.exp().mean().item()
                    if bc_loss is not None:
                        total_bc_loss += bc_loss.item()
                        bc_batches += 1
                num_batches += 1
            if (
                self.target_kl
                and self.target_kl > 0.0
                and epoch_batches > 0
                and (epoch_kl / epoch_batches) > self.target_kl
            ):
                early_stop = 1
                break

        self.total_updates += 1
        if num_batches == 0:
            return {}
        return {
            "pg_loss": total_pg_loss / num_batches,
            "vf_loss": total_vf_loss / num_batches,
            "entropy": total_ent / num_batches,
            "normalized_entropy": total_norm_ent / num_batches,
            "approx_kl": total_kl / num_batches,
            "clip_fraction": total_clip_frac / num_batches,
            "explained_variance": explained_variance,
            "mean_advantage": mean_advantage,
            "std_advantage": std_advantage,
            "invalid_action_count": invalid_action_count,
            "mean_chosen_action_prob": total_chosen_prob / num_batches,
            "bc_loss": total_bc_loss / bc_batches if bc_batches else 0.0,
            "bc_coef": self.bc_coef,
            "ent_coef": self.ent_coef,
            "lr": self.get_lr(),
            "early_stop": early_stop,
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        torch.save({
            "shared": self.shared.state_dict(),
            "policy_head": self.policy_head.state_dict(),
            "value_head": self.value_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_updates": self.total_updates,
            "lr": self.get_lr(),
            "ent_coef": self.ent_coef,
            "bc_coef": self.bc_coef,
        }, tmp)
        os.replace(tmp, path)

    def load(self, path: str, load_hparams: bool = False):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        try:
            self.shared.load_state_dict(ckpt["shared"])
            self.policy_head.load_state_dict(ckpt["policy_head"])
            self.value_head.load_state_dict(ckpt["value_head"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
        except RuntimeError:
            # Shape mismatch (e.g. OBS_SIZE changed) — start fresh weights
            # but keep total_updates so logs stay coherent.
            pass
        self.total_updates = ckpt.get("total_updates", 0)
        self._loaded_bc_coef = None
        if load_hparams:
            self.ent_coef = float(ckpt.get("ent_coef", self.ent_coef))
            if "bc_coef" in ckpt:
                self.bc_coef = float(ckpt["bc_coef"])
                self._loaded_bc_coef = self.bc_coef
            if "lr" in ckpt:
                self.set_lr(float(ckpt["lr"]))
