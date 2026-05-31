# Cars/shared/semantic_shaper.py
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class MiniLMTransitionShaper:
    """
    Semantic shaping based on cosine similarity between:
      1) planner-selected goal text
      2) observed transition text: before predicates + action + after predicates + changed predicates

    This is not a hand-coded reward table. The only environment-specific part is how states
    are converted into textual predicates, which already exists in MiniGridAdapter.
    """

    ACTION_NAMES = {
        0: "turn left",
        1: "turn right",
        2: "move forward",
        3: "pick up object",
        4: "drop carried object",
        5: "toggle or interact with object",
        6: "done",
    }

    GOAL_TEXT = {
        "explore": "explore the environment and discover useful objects",
        "find_key": "find the correct key needed for the task",
        "pickup_key": "pick up the correct key needed for the task",
        "drop_key": "drop the currently carried wrong object or wrong key",
        "find_door": "find the correct door relevant to the task",
        "open_door": "open the correct door and complete the task",
        "pickup_ball": "pick up the target ball",
        "pickup_box": "pick up the target box",
    }

    def __init__(
        self,
        model_path="/workspace/tools/all-MiniLM-L6-v2",
        device="cuda",
        gamma=0.99,
        eta_potential=0.05,
        eta_transition=0.02,
        clip_reward=0.05,
    ):
        self.device = device
        self.gamma = float(gamma)
        self.eta_potential = float(eta_potential)
        self.eta_transition = float(eta_transition)
        self.clip_reward = float(clip_reward)

        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_path, device=device)

        self._cache = {}

    def _embed(self, text: str) -> torch.Tensor:
        text = str(text)
        if text not in self._cache:
            emb = self.model.encode(
                [text],
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            self._cache[text] = emb.detach()
        return self._cache[text]

    def _cos(self, a_text: str, b_text: str) -> float:
        a = self._embed(a_text)
        b = self._embed(b_text)
        return float(F.cosine_similarity(a, b, dim=0).detach().cpu().item())

    def goal_to_text(self, goal: str) -> str:
        goal = str(goal or "explore")
        return self.GOAL_TEXT.get(goal, goal.replace("_", " "))

    def state_to_text(self, state, adapter=None) -> str:
        parts = []

        try:
            if adapter is not None:
                preds = adapter.extract_active_predicates(state)
                parts.extend([str(p).replace(":", " ") for p in preds])
        except Exception:
            pass

        try:
            visible = state.get("visible_objects", []) or []
            if visible:
                parts.append("visible objects: " + ", ".join(map(str, visible)))
        except Exception:
            pass

        try:
            carrying = state.get("carrying", []) or []
            if carrying:
                parts.append("carrying: " + ", ".join(map(str, carrying)))
            else:
                parts.append("carrying nothing")
        except Exception:
            pass

        for key in [
            "key_visible",
            "relevant_key_visible",
            "door_visible",
            "door_open",
            "door_locked",
            "has_key",
            "holding_relevant_key",
            "holding_wrong_key",
            "lava_visible",
        ]:
            try:
                if bool(state.get(key, False)):
                    parts.append(key.replace("_", " "))
            except Exception:
                pass

        if not parts:
            return "unknown state"

        return "; ".join(sorted(set(parts)))

    def transition_to_text(self, goal, prev_state, action, next_state, adapter=None) -> str:
        goal_text = self.goal_to_text(goal)

        try:
            a = int(np.asarray(action).reshape(-1)[0])
        except Exception:
            a = int(action)

        action_text = self.ACTION_NAMES.get(a, f"action {a}")

        before_text = self.state_to_text(prev_state, adapter)
        after_text = self.state_to_text(next_state, adapter)

        changed = []
        try:
            prev_preds = set(adapter.extract_active_predicates(prev_state)) if adapter is not None else set()
            next_preds = set(adapter.extract_active_predicates(next_state)) if adapter is not None else set()

            appeared = sorted(next_preds - prev_preds)
            disappeared = sorted(prev_preds - next_preds)

            if appeared:
                changed.append("became true: " + ", ".join(x.replace(":", " ") for x in appeared))
            if disappeared:
                changed.append("became false: " + ", ".join(x.replace(":", " ") for x in disappeared))
        except Exception:
            pass

        changed_text = "; ".join(changed) if changed else "no symbolic predicate changed"

        return (
            f"goal: {goal_text}. "
            f"before: {before_text}. "
            f"action: {action_text}. "
            f"after: {after_text}. "
            f"change: {changed_text}."
        )

    def potential(self, goal, state, adapter=None) -> float:
        goal_text = self.goal_to_text(goal)
        state_text = self.state_to_text(state, adapter)
        return self._cos(goal_text, state_text)

    def transition_alignment(self, goal, prev_state, action, next_state, adapter=None) -> float:
        goal_text = self.goal_to_text(goal)
        trans_text = self.transition_to_text(goal, prev_state, action, next_state, adapter)
        sim = self._cos(goal_text, trans_text)

        # cosine 可能为负，负值不宜直接当惩罚，否则语言模型噪声会伤害 PPO。
        return max(0.0, sim)

    def shaping_reward(self, goal, prev_state, action, next_state, adapter=None) -> tuple[float, dict]:
        phi_prev = self.potential(goal, prev_state, adapter)
        phi_next = self.potential(goal, next_state, adapter)

        # Potential-based term: gamma * Phi(s') - Phi(s)
        r_phi = self.gamma * phi_next - phi_prev

        # Transition semantic alignment term
        r_align = self.transition_alignment(goal, prev_state, action, next_state, adapter)

        r = self.eta_potential * r_phi + self.eta_transition * r_align
        r = float(np.clip(r, -self.clip_reward, self.clip_reward))

        diag = {
            "phi_prev": float(phi_prev),
            "phi_next": float(phi_next),
            "r_phi": float(r_phi),
            "r_align": float(r_align),
            "semantic_shape_reward": float(r),
        }
        return r, diag