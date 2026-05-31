# /workspace/Causal-aware_LLMs/agent/model/ppo_ad.py
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

from crafter_cars.agent.model.base import BaseModel
from crafter_cars.agent.impala_cnn import ImpalaCNN
from crafter_cars.agent.action_head import CategoricalActionHead
from crafter_cars.agent.mse_head import ScaledMSEHead
from crafter_cars.agent.torch_util import FanInInitReLULayer

# 和 PPOModel 一致：语言编码器
from crafter_cars.encoder import SbertEncoder


def _kl_logits(student_logits: th.Tensor, teacher_logits: th.Tensor, temperature: float = 1.0) -> th.Tensor:
    """KL( teacher || student ) distillation in logit space."""
    t = max(float(temperature), 1e-6)
    p = F.softmax(teacher_logits / t, dim=-1)
    log_q = F.log_softmax(student_logits / t, dim=-1)
    return F.kl_div(log_q, p, reduction="batchmean") * (t * t)


class PPOADModel(BaseModel):
    """
    适配你当前 RolloutStorage 的 PPOADModel：

    - act() 必须返回 latents / states / actions / log_probs / vpreds
    - encode() 接收 dict: {obs, text_obs_emd, goals_emd}（兼容 storage.insert() 的调用）
      也兼容仅传入 obs Tensor（aux 阶段/Buffer 只有 obs）
    - latent 维度固定为 2*hidsize（image hidsize + lang hidsize），
      对齐你 train.py 里 RolloutStorage(hidsize=config["hidsize"]*2)
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        hidsize: int,
        impala_kwargs: Dict = {},
        dense_init_norm_kwargs: Dict = {},
        action_head_kwargs: Dict = {},
        mse_head_kwargs: Dict = {},
        device: str = "cuda",
        sbert_path: Optional[str] = None,
        # aux 超参（和 algorithm/ppo_ad.py 对齐）
        contrastive_temp: float = 0.1,
        distill_temp: float = 1.0,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, device=device)
        self.device = device

        obs_shape = getattr(self.observation_space, "shape")
        num_actions = getattr(self.action_space, "n")

        # -------- Image encoder (same family as PPOModel) --------
        self.image_enc = ImpalaCNN(
            obs_shape,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            **impala_kwargs,
        )
        outsize = impala_kwargs.get("outsize")
        if outsize is None:
            raise ValueError("impala_kwargs must contain key 'outsize' (same as PPOModel).")

        self.linear = FanInInitReLULayer(
            outsize,
            hidsize,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )

        # -------- Language encoder (optional, but we keep latent dim = 2*hidsize) --------
        self.lang_goal_encoder: Optional[SbertEncoder] = None
        if sbert_path is not None:
            self.lang_goal_encoder = SbertEncoder(hidsize, device, path=sbert_path).to(self.device)

        # latent_dim 固定为 2*hidsize（对齐 storage.states 的维度初始化用 hidsize*2）
        self.base_hidsize = int(hidsize)
        self.latent_dim = int(hidsize) * 2
        self.state_dim = self.latent_dim  # storage.states 也是这个维度（你传的是 hidsize*2）

        # -------- Policy/Value heads condition on [latent, state] --------
        # 输入维度 = latent_dim + state_dim = 2*(2*hidsize) = 4*hidsize
        pi_in = self.latent_dim + self.state_dim

        self.pi_head = CategoricalActionHead(
            insize=pi_in,
            num_actions=num_actions,
            **action_head_kwargs,
        )
        self.vf_head = ScaledMSEHead(
            insize=pi_in,
            outsize=1,
            **mse_head_kwargs,
        )

        # -------- Aux: map image transition -> state space (latent_dim) --------
        # aux 阶段没有 text embedding，所以只能用图像差分，投影到 state space
        self.goal_delta_head = nn.Sequential(
            FanInInitReLULayer(hidsize, hidsize, layer_type="linear", **dense_init_norm_kwargs),
            nn.ReLU(inplace=True),
            FanInInitReLULayer(hidsize, self.state_dim, layer_type="linear", **dense_init_norm_kwargs),
        )

        self.contrastive_temp = float(contrastive_temp)
        self.distill_temp = float(distill_temp)

        self.to(self.device)

    # ---------------- Encode helpers ----------------
    def _encode_image(self, obs: th.Tensor) -> th.Tensor:
        x = self.image_enc(obs)
        x = self.linear(x)  # [B, hidsize]
        return x

    def encode(self, input: Dict[str, th.Tensor] | th.Tensor) -> th.Tensor:
        """
        返回 latents: [B, 2*hidsize]
        - 如果 input 是 dict 且包含 text_obs_emd / goals_emd 且存在 lang encoder，则拼接语言向量
        - 否则语言向量补零（保证维度始终为 2*hidsize，和 RolloutStorage 对齐）
        """
        if isinstance(input, dict):
            obs = input["obs"]
            text_obs_emd = input.get("text_obs_emd", None)
            goals_emd = input.get("goals_emd", None)
        else:
            obs = input
            text_obs_emd = None
            goals_emd = None

        x_img = self._encode_image(obs)  # [B, h]

        # 语言部分默认补 0，保证 latent_dim = 2*h
        if self.lang_goal_encoder is not None and text_obs_emd is not None and goals_emd is not None:
            # text_obs_emd/goals_emd 在你 storage 里是 token id 序列（shape: [B, L]）
            state_and_goal = torch.cat([text_obs_emd, goals_emd], dim=-1)
            x_lang = self.lang_goal_encoder(state_and_goal.to(x_img.device))
        else:
            x_lang = torch.zeros(x_img.shape[0], self.base_hidsize, device=x_img.device, dtype=x_img.dtype)

        latents = torch.cat([x_img, x_lang], dim=-1)  # [B, 2*h]
        return latents

    # ---------------- Forward / Act ----------------
    def forward(self, input: Dict[str, th.Tensor] | th.Tensor, states: Optional[th.Tensor] = None) -> Dict[str, th.Tensor]:
        """
        支持：
        - forward({"obs": obs, "states": states, "text_obs_emd":..., "goals_emd":...})
        - forward(obs, states)
        """
        if isinstance(input, dict):
            states = input.get("states", states)
            latents = self.encode(input)
        else:
            latents = self.encode(input)

        if states is None:
            # PPOAD 依赖 storage 算出来的 states；若没给则置 0（安全兜底）
            states = torch.zeros(latents.shape[0], self.state_dim, device=latents.device, dtype=latents.dtype)

        pi_latents = torch.cat([latents, states], dim=-1)
        vf_latents = pi_latents

        pi_logits = self.pi_head(pi_latents)
        vpreds = self.vf_head(vf_latents)

        return {
            "latents": latents,
            "states": states,
            "pi_latents": pi_latents,
            "vf_latents": vf_latents,
            "pi_logits": pi_logits,
            "vpreds": vpreds,
        }

    @th.no_grad()
    def act(self, input: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        """
        rollout 时使用（storage.insert 依赖 outputs['latents']）
        """
        assert not self.training
        outputs = self.forward(input)

        pi_logits = outputs["pi_logits"]
        actions = self.pi_head.sample(pi_logits)
        log_probs = self.pi_head.log_prob(pi_logits, actions)

        vpreds = outputs["vpreds"]
        vpreds = self.vf_head.denormalize(vpreds)

        outputs.update({"actions": actions, "log_probs": log_probs, "vpreds": vpreds})
        return outputs

    # ---------------- PPO loss ----------------
    def compute_losses(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        log_probs: th.Tensor,
        vtargs: th.Tensor,
        advs: th.Tensor,
        states: th.Tensor,
        text_obs_emd: Optional[th.Tensor] = None,
        goals_emd: Optional[th.Tensor] = None,
        clip_param: float = 0.2,
        **kwargs,
    ) -> Dict[str, th.Tensor]:
        # forward 需要把 text/goals 带进去（对齐你 storage.get_data_loader()）
        outputs = self.forward(
            {"obs": obs, "states": states, "text_obs_emd": text_obs_emd, "goals_emd": goals_emd}
        )

        pi_logits = outputs["pi_logits"]
        new_log_probs = self.pi_head.log_prob(pi_logits, actions)

        ratio = th.exp(new_log_probs - log_probs)
        ratio_clipped = th.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
        pi_loss = -th.min(advs * ratio, advs * ratio_clipped).mean()

        entropy = self.pi_head.entropy(pi_logits).mean()

        vpreds = outputs["vpreds"]
        vf_loss = self.vf_head.mse_loss(vpreds, vtargs).mean()

        return {"pi_loss": pi_loss, "vf_loss": vf_loss, "entropy": entropy}

    # ---------------- Aux: goal states ----------------
    @th.no_grad()
    def get_states(self, goal_obs: th.Tensor, goal_next_obs: th.Tensor) -> th.Tensor:
        """
        algorithm/ppo_ad.py 的 match 阶段需要：
          states = model.get_states(goal_obs, goal_next_obs)
        这里 aux 阶段没有 text embedding，因此用图像差分并投影到 state_dim。
        """
        z0 = self._encode_image(goal_obs)       # [B, h]
        z1 = self._encode_image(goal_next_obs)  # [B, h]
        delta = z1 - z0                         # [B, h]
        s = self.goal_delta_head(delta)         # [B, 2h]
        s = F.normalize(s, dim=-1)
        return s

    # ---------------- Aux: losses ----------------
    def compute_match_losses(
        self,
        anc_goal_obs: th.Tensor,
        anc_goal_next_obs: th.Tensor,
        pos_goal_obs: th.Tensor,
        pos_goal_next_obs: th.Tensor,
        neg_goal_obs: th.Tensor,
        neg_goal_next_obs: th.Tensor,
        obs: th.Tensor,
        old_states: th.Tensor,
        old_vtargs: th.Tensor,
        old_model: "PPOADModel",
        **kwargs,
    ) -> Dict[str, th.Tensor]:
        # contrastive: anc closer to pos than neg
        anc_s = self.get_states(anc_goal_obs, anc_goal_next_obs)
        pos_s = self.get_states(pos_goal_obs, pos_goal_next_obs)
        neg_s = self.get_states(neg_goal_obs, neg_goal_next_obs)

        temp = max(self.contrastive_temp, 1e-6)
        sim_pos = (F.normalize(anc_s, dim=-1) * F.normalize(pos_s, dim=-1)).sum(dim=-1) / temp
        sim_neg = (F.normalize(anc_s, dim=-1) * F.normalize(neg_s, dim=-1)).sum(dim=-1) / temp

        logits = th.stack([sim_pos, sim_neg], dim=1)  # [B,2]
        labels = th.zeros(logits.shape[0], dtype=th.long, device=logits.device)
        match_loss = F.cross_entropy(logits, labels)

        # distill policy/value to keep behavior close to old_model
        with th.no_grad():
            old_out = old_model.forward(obs, states=old_states)
            old_pi = old_out["pi_logits"]
            old_v = old_model.vf_head.denormalize(old_out["vpreds"])

        cur_out = self.forward(obs, states=old_states)
        cur_pi = cur_out["pi_logits"]
        cur_v = self.vf_head.denormalize(cur_out["vpreds"])

        pi_dist = _kl_logits(cur_pi, old_pi, temperature=self.distill_temp)
        vf_dist = F.mse_loss(cur_v, old_v)

        return {"match_loss": match_loss, "pi_dist": pi_dist, "vf_dist": vf_dist}

    def compute_pred_losses(
        self,
        anc_goal_obs: th.Tensor,
        anc_goal_next_obs: th.Tensor,
        pos_obs: th.Tensor,
        pos_actions: th.Tensor,
        pos_old_states: th.Tensor,
        pos_old_vtargs: th.Tensor,
        neg_obs: th.Tensor,
        neg_actions: th.Tensor,
        neg_old_states: th.Tensor,
        neg_old_vtargs: th.Tensor,
        old_model: "PPOADModel",
        **kwargs,
    ) -> Dict[str, th.Tensor]:
        # anc goal state
        anc_s = self.get_states(anc_goal_obs, anc_goal_next_obs)

        # pos/neg targets are stored old_states (already normalized in storage.insert)
        pos_t = F.normalize(pos_old_states.detach(), dim=-1)
        neg_t = F.normalize(neg_old_states.detach(), dim=-1)

        temp = max(self.contrastive_temp, 1e-6)
        sim_pos = (F.normalize(anc_s, dim=-1) * pos_t).sum(dim=-1) / temp
        sim_neg = (F.normalize(anc_s, dim=-1) * neg_t).sum(dim=-1) / temp

        logits = th.stack([sim_pos, sim_neg], dim=1)
        labels = th.zeros(logits.shape[0], dtype=th.long, device=logits.device)
        pred_loss = F.cross_entropy(logits, labels)

        # distill on pos_obs
        with th.no_grad():
            old_out = old_model.forward(pos_obs, states=pos_old_states)
            old_pi = old_out["pi_logits"]
            old_v = old_model.vf_head.denormalize(old_out["vpreds"])

        cur_out = self.forward(pos_obs, states=pos_old_states)
        cur_pi = cur_out["pi_logits"]
        cur_v = self.vf_head.denormalize(cur_out["vpreds"])

        pi_dist = _kl_logits(cur_pi, old_pi, temperature=self.distill_temp)
        vf_dist = F.mse_loss(cur_v, old_v)

        return {"pred_loss": pred_loss, "pi_dist": pi_dist, "vf_dist": vf_dist}
