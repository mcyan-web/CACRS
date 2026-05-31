from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque, defaultdict
from typing import Dict, Any, List, Optional, Tuple, Set
import re

@dataclass
class SkillRecord:
    name: str
    effect: str = ""
    preconditions: Dict[str, float] = field(default_factory=dict)
    success_count: int = 0
    fail_count: int = 0
    reuse_count: int = 0
    avg_cost: float = 0.0
    last_seen_step: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        if total <= 0:
            return 0.0
        return self.success_count / total

    def confidence(self) -> float:
        total = self.success_count + self.fail_count
        if total == 0:
            return 0.0
        sr = self.success_rate()
        support = min(1.0, total / 10.0)
        return sr * support

    def update(self, success: bool, cost: float, global_step: int) -> None:
        if success:
            self.success_count += 1
        else:
            self.fail_count += 1

        total = self.success_count + self.fail_count
        if total == 1:
            self.avg_cost = float(cost)
        else:
            self.avg_cost = ((self.avg_cost * (total - 1)) + float(cost)) / total

        self.last_seen_step = int(global_step)

    def mark_reused(self) -> None:
        self.reuse_count += 1


class SkillMemory:
    """
    Attempt-based skill memory with a DAG over successful skill transitions.

    Key fixes versus the old implementation:
      - statistics are updated per finalized attempt, NOT per environment step
      - active attempts are tracked per env
      - successful transitions can induce prerequisite edges in a DAG
      - retrieval can leverage DAG frontier information in addition to state match
    """

    def __init__(
        self,
        env_name: str,
        min_success_rate: float = 0.2,
        min_confidence: float = 0.1,
        history_size: int = 50,
        preserve_minigrid_colors: bool = False,
    ):
        self.env_name = env_name.lower()
        self.min_success_rate = float(min_success_rate)
        self.min_confidence = float(min_confidence)
        self.history_size = int(history_size)
        self.preserve_minigrid_colors = bool(preserve_minigrid_colors)

        self.skills: Dict[str, SkillRecord] = {}
        self.global_step: int = 0

        self._history: Dict[str, deque] = {}
        self._active_attempts: Dict[int, Dict[str, Any]] = {}
        self._last_success_goal_by_env: Dict[int, str] = {}

        self.parents: Dict[str, Set[str]] = defaultdict(set)
        self.children: Dict[str, Set[str]] = defaultdict(set)
        self.dep_success: Dict[Tuple[str, str], int] = defaultdict(int)
        self.dep_fail: Dict[Tuple[str, str], int] = defaultdict(int)
        self.dep_last_seen_step: Dict[Tuple[str, str], int] = defaultdict(int)
    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def _normalize_minigrid_predicate_for_skill(self, pred: Any) -> str:
        """
        Skill memory normally learns abstract MiniGrid skill schemas.

        However, ColoredDoorKey requires color binding:
        - blue_key opens blue_door
        - red_key should not be generalized as useful for blue_door

        Therefore when preserve_minigrid_colors=True, key and door predicates keep
        their concrete color. Other movable object colors may still be abstracted.
        """
        s = str(pred or "").strip()

        if self.env_name != "minigrid":
            return s

        # ------------------------------------------------------------
        # ColoredDoorKey mode:
        # preserve key and door colors.
        # ------------------------------------------------------------
        if self.preserve_minigrid_colors:
            # Keep these unchanged:
            #   object:key:visible(blue)
            #   state:agent:holding_key(blue)
            #   object:door:visible(blue)
            #   object:door:locked(blue)
            #   object:door:open(blue)

            # But still abstract ball / box colors to avoid unnecessary pollution.
            s = re.sub(
                r"^(object:(?:ball|box):visible)\([^)]+\)$",
                r"\1(any)",
                s,
            )

            s = re.sub(
                r"^(state:agent:holding_(?:ball|box))\([^)]+\)$",
                r"\1(any)",
                s,
            )

            return s

        # ------------------------------------------------------------
        # Default MiniGrid mode:
        # abstract colors for general skill schemas.
        # ------------------------------------------------------------

        # object:key:visible(green) -> object:key:visible(any)
        s = re.sub(
            r"^(object:key:visible)\([^)]+\)$",
            r"\1(any)",
            s,
        )

        # state:agent:holding_key(green) -> state:agent:holding_key(any)
        s = re.sub(
            r"^(state:agent:holding_key)\([^)]+\)$",
            r"\1(any)",
            s,
        )

        # object:door:visible(green) / locked / open -> (...any)
        s = re.sub(
            r"^(object:door:(?:visible|locked|open))\([^)]+\)$",
            r"\1(any)",
            s,
        )

        # 其他可搬运物体也可抽象，防止 ball/box 颜色污染
        s = re.sub(
            r"^(object:(?:ball|box):visible)\([^)]+\)$",
            r"\1(any)",
            s,
        )

        s = re.sub(
            r"^(state:agent:holding_(?:ball|box))\([^)]+\)$",
            r"\1(any)",
            s,
        )

        return s


    def _normalize_skill_predicates(self, preds: Any) -> List[str]:
        out = []
        for p in preds or []:
            q = self._normalize_minigrid_predicate_for_skill(p)
            if q and q not in out:
                out.append(q)
        return out
    def retrieve_goal_candidates(self, state: Dict[str, Any], topk: int = 2, adapter=None) -> List[str]:
        scored = []
        for skill_name, record in self.skills.items():
            score = self.support_score(state, skill_name, adapter=adapter)
            if score > 0.0:
                scored.append((skill_name, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in scored[:topk]]

    def support_score(self, state: Dict[str, Any], goal: str, adapter=None) -> float:
        record = self.skills.get(goal)
        if record is None:
            return 0.0

        if record.success_rate() < self.min_success_rate:
            return 0.0

        conf = record.confidence()
        if conf < self.min_confidence:
            return 0.0

        match = 0.0
        if adapter is not None and hasattr(adapter, "skill_match_score"):
            match = float(adapter.skill_match_score(state, goal))
        elif adapter is not None and hasattr(adapter, "goal_progress_signal"):
            try:
                match = float(adapter.goal_progress_signal(goal, state))
            except Exception:
                match = 0.0

        reuse_bonus = min(1.0, record.reuse_count / 10.0)
        low_cost_bonus = 1.0 / (1.0 + max(0.0, record.avg_cost))
        frontier_bonus = self._frontier_bonus(goal, state, adapter)

        return 0.40 * match + 0.25 * conf + 0.15 * frontier_bonus + 0.10 * reuse_bonus + 0.10 * low_cost_bonus


    def register_skill_schema(
        self,
        skill_id: str,
        effect: Optional[str] = None,
        preconditions: Optional[Dict[str, float]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> SkillRecord:
        """Register or update an executable skill schema.

        A skill is indexed by its executable id but organized around an effect
        predicate.  This keeps skill memory separate from the entity-level
        belief graph while allowing graph frontier predicates to retrieve
        executable candidates.
        """
        skill_id = str(skill_id)
        self._ensure_skill(skill_id)
        rec = self.skills[skill_id]
        if effect:
            rec.effect = str(effect)
        if preconditions:
            for k, v in preconditions.items():
                rec.preconditions[str(k)] = float(v)
        if meta:
            rec.meta.update(dict(meta))
        return rec

    def skills_for_effect(self, effect_predicate: str) -> List[SkillRecord]:
        effect_predicate = str(effect_predicate)
        exact = [r for r in self.skills.values() if r.effect == effect_predicate]
        if exact:
            return sorted(exact, key=lambda r: r.confidence(), reverse=True)
        # Backward compatible: legacy goals are treated as achieve(effect).
        legacy_id = self.effect_to_legacy_goal(effect_predicate)
        if legacy_id in self.skills:
            rec = self.skills[legacy_id]
            if not rec.effect:
                rec.effect = effect_predicate
            return [rec]
        return []

    def propose_skills_from_frontier(
        self,
        frontier_predicates: List[Dict[str, Any]],
        active_predicates: Optional[Set[str]] = None,
        adapter=None,
        topk: int = 8,
    ) -> List[Dict[str, Any]]:
        active = {
            self._normalize_minigrid_predicate_for_skill(p)
            for p in (active_predicates or set())
        }

        out: List[Dict[str, Any]] = []
        seen = set()

        for item in frontier_predicates or []:
            raw_effect = str(item.get("predicate") or item.get("effect") or "")
            effect = self._normalize_minigrid_predicate_for_skill(raw_effect)

            if not effect or effect in active:
                continue

            records = self.skills_for_effect(effect)

            if not records:
                goal = (
                    adapter.predicate_to_goal(effect)
                    if adapter is not None and hasattr(adapter, "predicate_to_goal")
                    else f"achieve({effect})"
                )

                records = [
                    self.register_skill_schema(
                        skill_id=goal,
                        effect=effect,
                        preconditions={},
                        meta={"unresolved_frontier": True},
                    )
                ]

            for rec in records:
                # 关键改动：
                # goal 可以来自历史 skill record，但 effect 必须来自当前 frontier。
                goal = rec.name

                if adapter is not None and hasattr(adapter, "predicate_to_goal"):
                    mapped_goal = adapter.predicate_to_goal(effect)
                    if mapped_goal:
                        goal = mapped_goal

                if goal in seen:
                    continue

                seen.add(goal)

                out.append({
                    "goal": goal,
                    "skill_id": rec.name,

                    # 关键：不要用 rec.effect 覆盖当前 frontier effect。
                    "effect": effect,

                    # skill schema 层做抽象化，防止颜色污染。
                    "preconditions": {
                        self._normalize_minigrid_predicate_for_skill(k): float(v)
                        for k, v in dict(rec.preconditions).items()
                    },

                    "source": item.get("source", "belief_frontier"),
                    "frontier_confidence": float(item.get("path_confidence", 0.0)),

                    # support_path 保留原始 belief graph 边，不要归一化。
                    # 因为后面对 belief_graph 做正负更新时要找到原图中的具体边。
                    "support_path": list(item.get("support_path", [])),

                    "target": item.get("target"),
                    "memory_confidence": rec.confidence(),
                    "expected_cost": rec.avg_cost,
                    "raw_frontier_effect": raw_effect,
                })

        out.sort(
            key=lambda x: (
                float(x.get("frontier_confidence", 0.0)),
                float(x.get("memory_confidence", 0.0)),
            ),
            reverse=True,
        )

        return out[: int(topk)]

    def planner_features(self, goal: str, candidate: Optional[Dict[str, Any]] = None, state: Optional[Dict[str, Any]] = None, adapter=None) -> Dict[str, float]:
        skill_id = str((candidate or {}).get("skill_id") or goal)
        rec = self.skills.get(skill_id) or self.skills.get(str(goal))
        if rec is None:
            return {"skill_confidence": 0.0, "skill_success_rate": 0.5, "skill_expected_cost": 0.0, "skill_dependency_satisfaction": 0.0, "skill_reuse_score": 0.0}
        dep_sat = 0.0
        if rec.preconditions:
            active = set()
            if adapter is not None and state is not None and hasattr(adapter, "extract_active_predicates"):
                try:
                    active = set(adapter.extract_active_predicates(state))
                except Exception:
                    active = set()
            weights = list(rec.preconditions.values())
            denom = sum(max(1e-6, float(w)) for w in weights) or 1.0
            dep_sat = sum(float(w) for p, w in rec.preconditions.items() if p in active) / denom
        else:
            dep_sat = 0.5
        attempts = rec.success_count + rec.fail_count
        alpha = rec.success_count + 1.0
        beta = rec.fail_count + 1.0
        posterior_mean = alpha / (alpha + beta)
        posterior_var = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1.0))

        return {
            "skill_confidence": float(rec.confidence()),
            "skill_success_rate": float(rec.success_rate() if attempts > 0 else 0.5),
            "skill_expected_cost": float(rec.avg_cost),
            "skill_dependency_satisfaction": float(dep_sat),
            "skill_reuse_score": min(1.0, float(rec.reuse_count) / 10.0),
            "skill_attempts": float(attempts),
            "skill_posterior_mean": float(posterior_mean),
            "skill_posterior_std": float(posterior_var ** 0.5),
        }

    def effect_to_legacy_goal(self, effect: str) -> str:
        e = str(effect)
        table = {
            "object:key:visible": "find_key",
            "state:agent:holding_key": "pickup_key",
            "object:door:visible": "find_door",
            "object:lava:visible": "avoid_lava",
            "object:door:open": "open_door",
            "terrain:tree:visible": "collect_wood",
            "item:wood:positive": "collect_wood",
            "terrain:water:visible": "collect_drink",
            "state:drink:restored": "collect_drink",
            "entity:cow:visible": "eat_cow",
            "state:food:restored": "eat_cow",
        }
        for prefix, goal in table.items():
            if e.startswith(prefix):
                return goal
        return f"achieve({e})"

    # ------------------------------------------------------------------
    # Attempt lifecycle
    # ------------------------------------------------------------------

    def start_attempt(
        self,
        env_index: int,
        goal: str,
        start_step: int,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not goal:
            return

        env_index = int(env_index)
        goal = str(goal)
        meta = dict(meta or {})

        cur = self._active_attempts.get(env_index)
        if cur is not None and cur.get("goal") == goal:
            return

        self._ensure_skill(goal)

        # New: schema grounding at attempt start.
        effect = meta.get("effect")
        prereqs = meta.get("prerequisites") or []

        effect = self._normalize_minigrid_predicate_for_skill(effect)
        prereqs = self._normalize_skill_predicates(prereqs)

        if effect:
            self.skills[goal].effect = str(effect)

        for p in prereqs:
            self.skills[goal].preconditions[str(p)] = max(
                self.skills[goal].preconditions.get(str(p), 0.0),
                float(meta.get("frontier_confidence", 0.5) or 0.5),
            )

        prev_success_goal = self._last_success_goal_by_env.get(env_index)

        self._active_attempts[env_index] = {
            "goal": goal,
            "start_step": int(start_step),
            "steps": 0,
            "cost": 0.0,
            "meta": {
                **meta,
                "prev_success_goal": prev_success_goal,
            },
        }

    def step_attempt(self, env_index: int, step_inc: int = 1, cost_inc: float = 1.0) -> None:
        cur = self._active_attempts.get(int(env_index))
        if cur is None:
            return
        cur["steps"] = int(cur.get("steps", 0)) + int(step_inc)
        cur["cost"] = float(cur.get("cost", 0.0)) + float(cost_inc)

    def end_attempt(
        self,
        env_index: int,
        success: bool,
        end_step: Optional[int] = None,
        reason: str = "terminated",
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        env_index = int(env_index)
        cur = self._active_attempts.get(env_index)
        if cur is None:
            return None

        goal = str(cur["goal"])
        self.global_step += 1
        self._ensure_skill(goal)

        record = {
            "goal": goal,
            "success": bool(success),
            "steps": int(cur.get("steps", 0)),
            "cost": float(cur.get("cost", 0.0)),
            "start_step": int(cur.get("start_step", -1)),
            "end_step": int(end_step) if end_step is not None else None,
            "env_index": env_index,
            "reason": str(reason),
            "meta": self._merge_meta(cur.get("meta", {}), meta or {}),
        }

        if goal not in self._history:
            self._history[goal] = deque(maxlen=self.history_size)
        self._history[goal].append(record)

        self.skills[goal].update(
            success=bool(success),
            cost=max(1.0, float(cur.get("cost", 0.0))),
            global_step=self.global_step,
        )

        if bool(success):
            meta_all = record["meta"]

            # 1) Learn / confirm effect predicate.
            effect = meta_all.get("effect")
            observed_effects = meta_all.get("observed_effects") or []

            if not effect and observed_effects:
                # Prefer non-risk positive effects.
                for e in observed_effects:
                    eff = e.get("effect") if isinstance(e, dict) else str(e)
                    if eff and "death_risk" not in eff:
                        effect = eff
                        break



            # 2) Learn predicate-level preconditions.
            # Prefer belief-supported prerequisites; otherwise use start predicates
            # only when they plausibly supported the observed effect.
            effect = self._normalize_minigrid_predicate_for_skill(effect)
            if effect:
                self.skills[goal].effect = str(effect)
            prereqs = self._normalize_skill_predicates(meta_all.get("prerequisites") or [])
            start_preds = self._normalize_skill_predicates(meta_all.get("start_predicates") or [])

            for p in prereqs:
                p = str(p)
                old = self.skills[goal].preconditions.get(p, 0.0)
                self.skills[goal].preconditions[p] = min(1.0, old + 0.10)

            # Weak adaptive support: predicates present at start of successful attempts
            # get small support, but only if an effect exists.
            if effect:
                for p in start_preds:
                    p = str(p)
                    if p == effect:
                        continue
                    old = self.skills[goal].preconditions.get(p, 0.0)
                    self.skills[goal].preconditions[p] = min(1.0, old + 0.02)

            # 3) Add skill-to-skill dependency only if semantically supported:
            # previous successful skill's effect appears in current preconditions.
            prev_success_goal = meta_all.get("prev_success_goal")
            if prev_success_goal and prev_success_goal in self.skills:
                prev_effect = self.skills[prev_success_goal].effect
                prev_effect = self._normalize_minigrid_predicate_for_skill(prev_effect)
                if prev_effect and prev_effect in self.skills[goal].preconditions:
                    self.add_dependency(prev_success_goal, goal, success=True)
            self._last_success_goal_by_env[env_index] = goal
        else:
            meta_all = record["meta"]
            # 当前 goal 失败时，弱惩罚已有 skill-DAG parents。
            # 注意：这里不直接删边，只记录失败证据，交给 prune_dependencies() 处理。
            for prereq in list(self.parents.get(goal, set())):
                self.update_dependency(prereq, goal, success=False)
            # 对失败 goal 的 precondition 做轻微衰减，避免无关谓词长期堆积。
            rec = self.skills[goal]
            for p in list(rec.preconditions.keys()):
                rec.preconditions[p] *= 0.98
                if rec.preconditions[p] < 0.03:
                    del rec.preconditions[p]
            
        try:
            self.prune_dependencies()
        except Exception:
            pass
        del self._active_attempts[env_index]
        return record

    def switch_goal(
        self,
        env_index: int,
        new_goal: str,
        step_idx: int,
        old_goal_success: bool = False,
        old_goal_reason: str = "replan_switch",
        old_goal_meta: Optional[Dict[str, Any]] = None,
        new_goal_meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        old_record = None
        cur = self._active_attempts.get(int(env_index))

        if cur is not None and cur.get("goal") != new_goal:
            old_record = self.end_attempt(
                env_index=env_index,
                success=bool(old_goal_success),
                end_step=step_idx,
                reason=old_goal_reason,
                meta=old_goal_meta,
            )

        self.start_attempt(
            env_index=env_index,
            goal=new_goal,
            start_step=step_idx,
            meta=new_goal_meta,
        )
        return old_record, self._active_attempts.get(int(env_index))

    def finalize_env(
        self,
        env_index: int,
        success: bool = False,
        end_step: Optional[int] = None,
        reason: str = "episode_end",
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        return self.end_attempt(
            env_index=env_index,
            success=success,
            end_step=end_step,
            reason=reason,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # DAG maintenance
    # ------------------------------------------------------------------

    def add_dependency(self, prereq: str, goal: str, success: bool = True) -> bool:
        prereq = str(prereq)
        goal = str(goal)

        if not prereq or not goal or prereq == goal:
            return False

        self._ensure_skill(prereq)
        self._ensure_skill(goal)

        edge = (prereq, goal)

        if bool(success):
            self.dep_success[edge] += 1
        else:
            self.dep_fail[edge] += 1

        self.dep_last_seen_step[edge] = int(self.global_step)

        if prereq in self.parents.get(goal, set()):
            return True

        if self._would_create_cycle(prereq, goal):
            return False

        self.parents[goal].add(prereq)
        self.children[prereq].add(goal)

        return True

    # ------------------------------------------------------------------
    # Backward-compatible wrapper
    # ------------------------------------------------------------------

    def update_from_transition(
        self,
        prev_state: Dict[str, Any],
        goal: str,
        action: Any,
        next_state: Dict[str, Any],
        success: bool,
        cost: float = 1.0,
        env_index: int = 0,
    ) -> None:
        """
        Backward-compatible wrapper.

        IMPORTANT:
          This no longer accumulates one failure per step.
          It only finalizes an active attempt when success=True.
        """
        if not goal:
            return
        if not self.has_active_attempt(env_index):
            self.start_attempt(env_index=env_index, goal=goal, start_step=-1, meta={"phase": "legacy"})
        self.step_attempt(env_index=env_index, step_inc=1, cost_inc=cost)
        if bool(success):
            self.end_attempt(env_index=env_index, success=True, reason="legacy_success")

    def mark_reused(self, goal: str) -> None:
        if goal in self.skills:
            self.skills[goal].mark_reused()

    # ------------------------------------------------------------------
    # Queries / stats
    # ------------------------------------------------------------------

    def has_active_attempt(self, env_index: int) -> bool:
        return int(env_index) in self._active_attempts

    def active_goal(self, env_index: int) -> Optional[str]:
        cur = self._active_attempts.get(int(env_index))
        if cur is None:
            return None
        return cur.get("goal")

    def summary(self) -> Dict[str, Any]:
        if not self.skills:
            return {
                "num_skills": 0,
                "mean_success_rate": 0.0,
                "mean_confidence": 0.0,
                "num_active_attempts": len(self._active_attempts),
                "num_dag_edges": 0,
            }

        records = list(self.skills.values())
        mean_sr = sum(x.success_rate() for x in records) / len(records)
        mean_conf = sum(x.confidence() for x in records) / len(records)
        num_edges = sum(len(v) for v in self.children.values())
        dep_confs = []
        for prereq, children in self.children.items():
            for goal in children:
                dep_confs.append(self.dependency_confidence(prereq, goal))
        successful_records = [x for x in records if x.success_count > 0]
        mean_sr_successful = (
            sum(x.success_rate() for x in successful_records) / len(successful_records)
            if successful_records else 0.0
        )

        return {
            "num_skills": len(records),
            "num_successful_skills": len(successful_records),
            "mean_success_rate": mean_sr,
            "mean_success_rate_successful_only": mean_sr_successful,
            "mean_confidence": mean_conf,
            "num_active_attempts": len(self._active_attempts),
            "num_dag_edges": num_edges,
            "mean_dag_confidence": sum(dep_confs) / len(dep_confs) if dep_confs else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_skill(self, goal: str) -> None:
        if goal not in self.skills:
            self.skills[goal] = SkillRecord(name=goal)

    def _frontier_bonus(self, goal: str, state: Dict[str, Any], adapter=None) -> float:
        parents = list(self.parents.get(goal, set()))
        if len(parents) == 0:
            return 0.5
        if adapter is None or not hasattr(adapter, "goal_satisfied"):
            return 0.0
        sat = 0
        for p in parents:
            try:
                if bool(adapter.goal_satisfied(p, state)):
                    sat += 1
            except Exception:
                pass
        return float(sat) / float(len(parents))

    def _would_create_cycle(self, prereq: str, goal: str) -> bool:
        stack = [goal]
        visited = set()
        while stack:
            cur = stack.pop()
            if cur == prereq:
                return True
            if cur in visited:
                continue
            visited.add(cur)
            stack.extend(self.children.get(cur, set()))
        return False

    def _merge_meta(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        out.update(a or {})
        out.update(b or {})
        return out
    def update_dependency(self, prereq: str, goal: str, success: bool) -> None:
        prereq = str(prereq)
        goal = str(goal)

        if not prereq or not goal or prereq == goal:
            return

        edge = (prereq, goal)

        if bool(success):
            self.dep_success[edge] += 1
        else:
            self.dep_fail[edge] += 1

        self.dep_last_seen_step[edge] = int(self.global_step)


    def dependency_confidence(self, prereq: str, goal: str) -> float:
        edge = (str(prereq), str(goal))
        s = int(self.dep_success.get(edge, 0))
        f = int(self.dep_fail.get(edge, 0))

        # Beta(1,1) posterior mean
        return float((s + 1.0) / (s + f + 2.0))


    def prune_dependencies(
        self,
        min_confidence: float = 0.25,
        min_support: int = 4,
        stale_after: int = 2000,
    ) -> None:
        """
        Prune weak or stale skill-DAG dependencies.

        A dependency A -> B is removed only when:
        1. it has enough observations and low confidence, or
        2. it is very stale and weakly supported.
        """
        all_edges = set(self.dep_success.keys()) | set(self.dep_fail.keys())

        for edge in list(all_edges):
            prereq, goal = edge

            s = int(self.dep_success.get(edge, 0))
            f = int(self.dep_fail.get(edge, 0))
            support = s + f
            conf = self.dependency_confidence(prereq, goal)
            last_seen = int(self.dep_last_seen_step.get(edge, 0))
            stale = (int(self.global_step) - last_seen) > int(stale_after)

            should_prune = False

            if support >= int(min_support) and conf < float(min_confidence):
                should_prune = True

            if stale and support < int(min_support) and conf < 0.5:
                should_prune = True

            if should_prune:
                self.parents[goal].discard(prereq)
                self.children[prereq].discard(goal)

                self.dep_success.pop(edge, None)
                self.dep_fail.pop(edge, None)
                self.dep_last_seen_step.pop(edge, None)