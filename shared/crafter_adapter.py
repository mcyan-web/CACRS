from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
import re


class CrafterAdapter:
    """
    Minimal enhanced Crafter adapter for:
      - planner goal scoring
      - belief graph node mapping
      - shaping reward support
      - transition-level success signal
    """

    DEFAULT_GOAL = "explore"

    def build_semantic_state(
        self,
        item: Dict[str, Any],
        achievement: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        inv_status = item.get("inv_status_backup", {}) or {}
        text_obs = item.get("text_obs_backup", "") or item.get("text_obs", "") or ""

        inventory_text = inv_status.get("inv", "") or ""
        status_text = inv_status.get("status", "") or ""

        visible_objects = self._extract_visible_objects(text_obs)
        inventory_items = self._parse_counts(inventory_text)
        status_values = self._parse_counts(status_text)
        past_goal = item.get("goal", None)
        state = {
            "env_name": "crafter",
            "text_obs": text_obs,
            "inventory_text": inventory_text,
            "status_text": status_text,
            "achievements": achievement or {},
            "obs": item.get("obs", None),
            "raw": item,
            "visible_objects": visible_objects,
            "inventory_items": inventory_items,
            "status_values": status_values,
            "past_goal": past_goal,
            "recent_progress_window": item.get("recent_progress_window", []),
        }

        return state

    # ------------------------------------------------------------------
    # candidate goals
    # ------------------------------------------------------------------
    def skill_match_score(self, state: Dict[str, Any], goal: str) -> float:
        goal = self._normalize_goal(goal) or ""

        visible = state.get("visible_objects", [])
        inventory = state.get("inventory_items", {})
        achievements = state.get("achievements", {})

        if achievements.get(goal, 0) > 0:
            return 1.0

        if goal.startswith("collect_"):
            obj = goal.replace("collect_", "")
            if goal == "collect_drink":
                return 0.7 if "water" in visible else 0.1
            if inventory.get(obj, 0) > 0:
                return 1.0
            mapped_obj = self._map_collect_goal_to_visible_object(obj)
            if mapped_obj in visible:
                return 0.7
            return 0.1

        if goal.startswith("place_"):
            obj = goal.replace("place_", "")
            if obj in visible:
                return 1.0
            return 0.2

        if goal.startswith("make_") or goal.startswith("defeat_"):
            return 0.3 if achievements.get(goal, 0) <= 0 else 1.0

        if goal in ("eat_cow", "eat_plant"):
            target = "cow" if goal == "eat_cow" else "plant"
            if target in visible:
                return 0.7
            return 0.1

        if goal in ("survive", "explore"):
            return 0.2

        return 0.0
    def enumerate_goal_candidates(
        self,
        state: Dict[str, Any],
        llm_goals: List[str],
        mem_goals: List[str],
        belief_graph=None,
        skill_memory=None,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        seen = set()
        if belief_graph is not None and hasattr(belief_graph, "frontier_to_target"):
            try:
                active = self.extract_active_predicates(state)
                targets = self.extract_task_objectives(state)
                frontier_items = belief_graph.frontier_to_target(active, targets, max_depth=3, min_conf=0.30)
                if skill_memory is not None and hasattr(skill_memory, "propose_skills_from_frontier"):
                    candidates.extend(skill_memory.propose_skills_from_frontier(frontier_items, active_predicates=set(active), adapter=self, topk=8))
                else:
                    for item in frontier_items:
                        g = self.predicate_to_goal(str(item.get("predicate", "")))
                        if g:
                            candidates.append({"goal": g, "source": item.get("source", "belief_frontier"), "effect": item.get("predicate"), "frontier_confidence": item.get("path_confidence", 0.0), "support_path": item.get("support_path", [])})
            except Exception:
                pass

        for g in llm_goals or []:
            g = self.canonicalize_goal(g)
            if g:
                candidates.append({"goal": g, "source": "llm"})

        for g in mem_goals or []:
            g = self.canonicalize_goal(g)
            if g:
                candidates.append({"goal": g, "source": "memory"})

        # No hand-coded object -> goal affordance table here.  Environment
        # observations enter as active predicates and candidates are derived via
        # belief-graph frontier plus skill-memory binding.

        if not candidates:
            candidates.append({"goal": self.default_goal(state), "source": "default"})

        seen = set()
        deduped = []
        for item in candidates:
            goal = item["goal"]
            if goal in seen:
                continue
            if self.goal_satisfied(goal, state):
                continue
            seen.add(goal)
            deduped.append(item)

        return deduped

    # ------------------------------------------------------------------
    # planner-facing signals
    # ------------------------------------------------------------------

    def goal_satisfied(self, goal: str, state: Dict[str, Any]) -> bool:
        goal = self._normalize_goal(goal) or ""

        achievements = state["achievements"]
        inventory = state["inventory_items"]
        visible = state["visible_objects"]

        if achievements.get(goal, 0) > 0:
            return True

        if goal.startswith("collect_"):
            obj = goal.replace("collect_", "")
            if goal == "collect_drink":
                return achievements.get("collect_drink", 0) > 0
            return inventory.get(obj, 0) > 0

        if goal.startswith("place_"):
            obj = goal.replace("place_", "")
            return obj in visible

        if goal in ("eat_cow", "eat_plant"):
            return achievements.get(goal, 0) > 0

        if goal.startswith("make_") or goal.startswith("defeat_"):
            return achievements.get(goal, 0) > 0

        if goal == "survive":
            return False

        if goal == "explore":
            return False

        return False

    def goal_progress_signal(self, goal: str, state: Dict[str, Any]) -> float:
        goal = self._normalize_goal(goal) or ""
        achievements = state["achievements"]
        inventory = state["inventory_items"]

        if achievements.get(goal, 0) > 0:
            return 1.0

        if goal.startswith("collect_"):
            obj = goal.replace("collect_", "")
            return 1.0 if inventory.get(obj, 0) > 0 else 0.0

        if goal.startswith("make_"):
            return 1.0 if achievements.get(goal, 0) > 0 else 0.0

        if goal.startswith("place_"):
            obj = goal.replace("place_", "")
            visible = state["visible_objects"]
            return 1.0 if obj in visible else 0.0

        if goal in ("eat_cow", "eat_plant", "collect_drink"):
            return 1.0 if achievements.get(goal, 0) > 0 else 0.0

        if goal == "survive":
            return 0.0

        if goal == "explore":
            return 0.0

        return 0.0

    def goal_progress_delta(self, goal: str, state: Dict[str, Any]) -> Optional[float]:
        recent = state.get("recent_progress_window", [])
        if not isinstance(recent, list) or len(recent) < 2:
            return None
        try:
            return float(recent[-1] - recent[0])
        except Exception:
            return None

    def goal_distance_estimate(self, goal: str, state: Dict[str, Any]) -> float:
        prog = self.goal_progress_signal(goal, state)
        return max(1.0, 5.0 * (1.0 - prog))

    def goal_novelty(self, goal: str, state: Dict[str, Any]) -> float:
        goal = self._normalize_goal(goal) or ""
        achievements = state["achievements"]
        if achievements.get(goal, 0) > 0:
            return 0.0
        return 1.0

    def infer_support_path(
        self,
        goal: str,
        state: Dict[str, Any],
        belief_graph=None,
    ) -> List[Tuple[str, str]]:
        if belief_graph is None:
            return []

        effect = self.goal_effect_predicate(goal, state)
        if not effect:
            return []

        incoming = belief_graph.incoming_edges(effect)
        incoming = sorted(
            incoming,
            key=lambda e: belief_graph.edge_confidence(e[0], e[1]),
            reverse=True,
        )
        return incoming[:3]

    # ------------------------------------------------------------------
    # NEW: belief/shaping support
    # ------------------------------------------------------------------

    def active_belief_nodes(self, state: Dict[str, Any]) -> List[str]:
        nodes: List[str] = []

        inventory = state.get("inventory_items", {}) or {}
        visible = state.get("visible_objects", []) or []
        achievements = state.get("achievements", {}) or {}

        for obj in visible:
            nodes.append(f"visible:{obj}")
            nodes.append(f"event:see_{obj}")

        for item, cnt in inventory.items():
            if cnt > 0:
                nodes.append(f"inventory:{item}")

        for ach, val in achievements.items():
            try:
                if int(val) > 0:
                    nodes.append(f"achievement:{self._normalize_goal(ach)}")
            except Exception:
                pass

        nodes.append("state:crafter")

        seen = set()
        deduped = []
        for n in nodes:
            if n not in seen:
                seen.add(n)
                deduped.append(n)

        return deduped

    def goal_node(self, goal: str) -> str:
        """
        将 planner goal 映射到 belief/shaping 所用的目标节点。
        """
        goal = self._normalize_goal(goal) or self.DEFAULT_GOAL
        return f"goal:{goal}"

    def transition_success_signal(
        self,
        goal: str,
        prev_state: Dict[str, Any],
        next_state: Dict[str, Any],
    ) -> float:
        """
        只基于前后状态差分和 achievement 增量判断本步是否真实推进/完成目标。
        不使用人工 prerequisite 启发。
        """
        goal = self._normalize_goal(goal) or ""

        prev_inv = prev_state.get("inventory_items", {}) or {}
        next_inv = next_state.get("inventory_items", {}) or {}

        prev_vis = set(prev_state.get("visible_objects", []) or [])
        next_vis = set(next_state.get("visible_objects", []) or [])

        prev_ach = prev_state.get("achievements", {}) or {}
        next_ach = next_state.get("achievements", {}) or {}

        prev_status = prev_state.get("status_values", {}) or {}
        next_status = next_state.get("status_values", {}) or {}

        def ach_inc(name: str) -> bool:
            return int(next_ach.get(name, 0)) > int(prev_ach.get(name, 0))

        # collect_* 以库存或状态真实增长为准
        if goal.startswith("collect_"):
            obj = goal.replace("collect_", "")

            if goal == "collect_drink":
                prev_drink = int(prev_status.get("drink", 0))
                next_drink = int(next_status.get("drink", 0))
                if next_drink > prev_drink:
                    return 1.0
                return 0.0

            prev_cnt = int(prev_inv.get(obj, 0))
            next_cnt = int(next_inv.get(obj, 0))
            if next_cnt > prev_cnt:
                return 1.0
            return 0.0

        # eat_cow: food +6, achievement eat_cow 增长, 且前后视野里出现过 cow
        if goal == "eat_cow":
            prev_food = int(prev_status.get("food", 0))
            next_food = int(next_status.get("food", 0))
            saw_cow = ("cow" in prev_vis) or ("cow" in next_vis)
            if (next_food - prev_food) == 6 and ach_inc("eat_cow") and saw_cow:
                return 1.0
            return 0.0

        # eat_plant: food +4, achievement eat_plant 增长, 且前后视野里出现过 plant
        if goal == "eat_plant":
            prev_food = int(prev_status.get("food", 0))
            next_food = int(next_status.get("food", 0))
            saw_plant = ("plant" in prev_vis) or ("plant" in next_vis)
            if (next_food - prev_food) == 4 and ach_inc("eat_plant") and saw_plant:
                return 1.0
            return 0.0

        # make_* / defeat_* / 其他 achievement 型目标：以 achievement 增长为准
        if goal.startswith("make_") or goal.startswith("defeat_"):
            if ach_inc(goal):
                return 1.0
            return 0.0

        # place_*：仅以对象新出现为准，不加材料消耗启发
        if goal.startswith("place_"):
            obj = goal.replace("place_", "")
            if obj in next_vis and obj not in prev_vis:
                return 1.0
            return 0.0

        # survive / explore 不做人为 transition 奖励
        if goal in ("survive", "explore"):
            return 0.0

        # 兜底：achievement 增长
        if ach_inc(goal):
            return 1.0

        return 0.0

    # ------------------------------------------------------------------
    # transition -> evidence
    # ------------------------------------------------------------------
    def goal_effect_predicate(self, goal: str, state: Dict[str, Any]) -> Optional[str]:
        """
        Map executable Crafter goal names to entity-level effect predicates.
        This is not a curriculum. It only states what observable predicate
        the skill is expected to change.
        """
        goal = self._normalize_goal(goal) or ""

        if goal == "collect_wood":
            return self._pred("item", "wood", "positive")

        if goal == "collect_stone":
            return self._pred("item", "stone", "positive")

        if goal == "collect_coal":
            return self._pred("item", "coal", "positive")

        if goal == "collect_iron":
            return self._pred("item", "iron", "positive")

        if goal == "collect_diamond":
            return self._pred("item", "diamond", "positive")

        if goal == "collect_drink":
            return self._pred("state", "drink", "sufficient")

        if goal == "eat_cow" or goal == "eat_plant":
            return self._pred("state", "food", "sufficient")

        if goal == "place_table":
            return self._pred("object", "table", "visible")

        if goal == "place_furnace":
            return self._pred("object", "furnace", "visible")

        if goal.startswith("make_"):
            item = goal.replace("make_", "")
            return self._pred("item", item, "positive")

        if goal.startswith("defeat_"):
            return self._pred("achievement", goal, "done")

        if goal.startswith("collect_") or goal.startswith("place_"):
            return self._pred("achievement", goal, "done")

        if goal == "survive":
            return self._pred("state", "survival", "maintained")

        if goal == "explore":
            return None

        return None
    def event_to_edge_evidence(
            self,
            prev_state: Dict[str, Any],
            action: Any,
            next_state: Dict[str, Any],
            goal: str,
        ) -> List[Dict[str, Any]]:
        """
        Entity-level transition evidence only.

        Belief graph stores:
        entity/state/item/terrain/achievement -> entity/state/item/terrain/achievement

        Prevents:
        event:* -> goal:*
        inventory:* -> goal:*
        goal:* -> goal:*
        """
        evidence: List[Dict[str, Any]] = []

        # 原来的 prev_active / next_active 转 set
        prev_active_set = set(self.extract_active_predicates(prev_state))
        next_active_set = set(self.extract_active_predicates(next_state))

        # 出现的新 predicates
        appeared = sorted(next_active_set - prev_active_set)
        max_sources_per_effect = 8

        # -----------------------------
        # 1) Generic entity transition evidence
        # -----------------------------
        for eff in appeared:
            # 过滤 prev_active，只保留明确实体 / 状态 / 成就
            valid_sources = [
                s for s in prev_active_set
                if s.startswith(("item:", "entity:", "object:", "state:", "achievement:"))
            ][:max_sources_per_effect]

            for src in valid_sources:
                if src == eff:
                    continue

                # 可以加过滤：防止 lava 等共现边滥用
                # 这里可按需要加 if self.is_valid_crafter_edge(src, eff)
                valid = True  # 如果需要可加自定义逻辑

                if not valid:
                    continue

                relation = "enables"
                weight = 0.03

                if "health:low" in eff or "risk" in eff or "lava" in src:
                    relation = "risks"
                    weight = 0.25
                elif "food" in eff or "drink" in eff or "health" in eff:
                    relation = "restores"
                    weight = 0.20
                elif eff.startswith("item:"):
                    relation = "provides"
                    weight = 0.20
                elif eff.startswith("achievement:"):
                    relation = "enables"
                    weight = 0.30

                evidence.append({
                    "edge": (src, eff),
                    "success": True,
                    "weight": weight,
                    "meta": {
                        "relation": relation,
                        "level": "entity_transition",
                        "goal_context": str(goal),
                    },
                })

        # -----------------------------
        # 2) 当前 goal 满足时，连接 active predicates 到 goal effect
        # -----------------------------
        effect = self.goal_effect_predicate(goal, next_state)
        if effect and self.goal_satisfied(goal, next_state):
            # 兼容 list / set 并集
            all_sources = prev_active_set | next_active_set
            # 限制每个 effect 最多 8 个源
            all_sources = sorted([s for s in all_sources
                                if s.startswith(("item:", "entity:", "object:", "state:", "achievement:"))][:max_sources_per_effect])

            for src in all_sources:
                if src == effect:
                    continue
                evidence.append({
                    "edge": (src, effect),
                    "success": True,
                    "weight": 0.35,
                    "meta": {
                        "relation": "enables",
                        "level": "skill_effect_support",
                        "goal_context": str(goal),
                    },
                })

        return evidence

    def default_goal(self, state: Dict[str, Any]) -> str:
        return self.DEFAULT_GOAL

    def policy_input(self, obs: Any, goal: str) -> Any:
        return obs


    # ------------------------------------------------------------------
    # Shared predicate graph interface
    # ------------------------------------------------------------------

    def _pred(self, domain: str, name: str, attr: str, *args) -> str:
        return f"{domain}:{name}:{attr}({','.join(str(a) for a in args)})"

    def extract_active_predicates(self, state: Dict[str, Any]) -> List[str]:
        nodes: List[str] = []
        inventory = state.get("inventory_items", {}) or {}
        visible = state.get("visible_objects", []) or []
        status = state.get("status_values", {}) or {}
        achievements = state.get("achievements", {}) or {}
        terrain_names = {"tree", "stone", "coal", "iron", "diamond", "water", "lava", "grass", "path"}
        entity_names = {"cow", "plant", "zombie", "skeleton"}
        object_names = {"table", "furnace"}
        for obj in visible:
            if obj in terrain_names:
                nodes.append(self._pred("terrain", obj, "visible"))
            elif obj in entity_names:
                nodes.append(self._pred("entity", obj, "visible"))
            elif obj in object_names:
                nodes.append(self._pred("object", obj, "visible"))
            else:
                nodes.append(self._pred("object", obj, "visible"))
        for item, cnt in inventory.items():
            try:
                if int(cnt) > 0:
                    nodes.append(self._pred("item", item, "positive"))
            except Exception:
                pass
        for name in ["food", "drink", "health", "energy"]:
            try:
                val = int(status.get(name, 0))
                if val <= 3:
                    nodes.append(self._pred("state", name, "low"))
                elif val >= 7:
                    nodes.append(self._pred("state", name, "sufficient"))
            except Exception:
                pass
        for ach, val in achievements.items():
            try:
                if int(val) > 0:
                    nodes.append(self._pred("achievement", self._normalize_goal(ach) or ach, "done"))
            except Exception:
                pass
        return list(dict.fromkeys(nodes))

    def extract_task_objectives(self, state: Dict[str, Any], reward_info=None) -> List[str]:
        # Open-ended Crafter uses unsatisfied achievements/effects as terminal
        # predicates.  This is a reward/achievement interface, not a hand-coded
        # curriculum over subgoals.
        achievements = state.get("achievements", {}) or {}
        targets: List[str] = []
        for ach, val in achievements.items():
            try:
                if int(val) <= 0:
                    g = self._normalize_goal(ach) or str(ach)
                    targets.append(self._pred("achievement", g, "done"))
            except Exception:
                pass
        if not targets:
            targets = [self._pred("state", "survival", "maintained")]
        return targets[:8]

    def predicate_to_goal(self, predicate: str) -> Optional[str]:
        p = str(predicate)
        if p.startswith("terrain:tree:visible") or p.startswith("item:wood:positive"):
            return "collect_wood"
        if p.startswith("terrain:water:visible") or p.startswith("state:drink:restored"):
            return "collect_drink"
        if p.startswith("entity:cow:visible") or p.startswith("state:food:restored"):
            return "eat_cow"
        if p.startswith("entity:plant:visible"):
            return "eat_plant"
        if p.startswith("object:table:visible"):
            return "place_table"
        if p.startswith("item:wood_pickaxe:positive") or p.startswith("achievement:make_wood_pickaxe"):
            return "make_wood_pickaxe"
        if p.startswith("item:stone:positive") or p.startswith("achievement:collect_stone"):
            return "collect_stone"
        if p.startswith("achievement:"):
            core = p.split(":", 1)[1].split(":", 1)[0]
            return self.canonicalize_goal(core)
        if "lava" in p or "risk" in p:
            return "survive"
        return None

    def extract_transition_events(self, prev_state: Dict[str, Any], action: Any, next_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        prev = set(self.extract_active_predicates(prev_state))
        nxt = set(self.extract_active_predicates(next_state))
        return [{"effect": p, "type": "positive"} for p in sorted(nxt - prev)]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def infer_goal_preconditions(
        self,
        goal: str,
        state: Dict[str, Any],
        belief_graph=None,
        min_conf: float = 0.30,
        max_items: int = 8,
    ) -> List[str]:
        effect = self.goal_effect_predicate(goal, state)
        if not effect or belief_graph is None:
            return []

        active = set(self.extract_active_predicates(state))
        incoming = belief_graph.incoming_edges(effect)

        scored = []
        for u, v in incoming:
            if v != effect:
                continue
            if u not in active:
                continue
            try:
                conf = float(belief_graph.edge_confidence(u, v))
            except Exception:
                conf = 0.0
            if conf >= min_conf:
                scored.append((u, conf))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [u for u, _ in scored[:max_items]]
    def goal_prerequisites(
        self,
        goal: str,
        state: Dict[str, Any],
        belief_graph=None,
    ) -> List[str]:
        """
        Backward-compatible name.

        Now returns predicate-level prerequisites, not goal-level prerequisites.
        """
        return self.infer_goal_preconditions(
            goal=goal,
            state=state,
            belief_graph=belief_graph,
        )

    def canonicalize_goal(self, goal: Optional[str]) -> Optional[str]:
        g = self._normalize_goal(goal)
        if not g:
            return None
        alias_map = {
            "chop_tree": "collect_wood",
            "cut_tree": "collect_wood",
            "get_wood": "collect_wood",
            "gather_wood": "collect_wood",
            "mine_stone": "collect_stone",
            "get_stone": "collect_stone",
            "gather_stone": "collect_stone",
            "mine_coal": "collect_coal",
            "get_coal": "collect_coal",
            "mine_iron": "collect_iron",
            "get_iron": "collect_iron",
            "mine_diamond": "collect_diamond",
            "get_diamond": "collect_diamond",
            "drink_water": "collect_drink",
            "drink": "collect_drink",
            "collect_water": "collect_drink",
            "gather_drink": "collect_drink",
            "craft_wood_pickaxe": "make_wood_pickaxe",
            "craft_stone_pickaxe": "make_stone_pickaxe",
            "craft_iron_pickaxe": "make_iron_pickaxe",
            "craft_wood_sword": "make_wood_sword",
            "craft_stone_sword": "make_stone_sword",
            "craft_iron_sword": "make_iron_sword",
            "place_crafting_table": "place_table",
            "crafting_table": "place_table",
            "place_crafting_bench": "place_table",
            "sleep": "wake_up",
        }
        return alias_map.get(g, g)

    def canonicalize_belief_edge(self, edge):
        """
        Canonicalize proposed belief edge.

        Accepted:
            (src, dst)
            (src, dst, relation)
            {"source": src, "target": dst, "relation": relation}
            {"src": src, "dst": dst, "relation": relation}

        Returns:
            (src, dst, relation) or None
        """
        if edge is None:
            return None

        if isinstance(edge, dict):
            u = edge.get("source", edge.get("src", None))
            v = edge.get("target", edge.get("dst", None))
            relation = edge.get("relation", "related")

        elif isinstance(edge, (tuple, list)):
            if len(edge) == 2:
                u, v = edge
                relation = "related"
            elif len(edge) == 3:
                u, v, relation = edge
            else:
                return None

        else:
            return None

        u = str(u).strip() if u is not None else ""
        v = str(v).strip() if v is not None else ""
        relation = str(relation).strip() if relation is not None else "related"

        alias = {
            "tree": self._pred("terrain", "tree", "visible"),
            "wood": self._pred("item", "wood", "positive"),
            "table": self._pred("object", "table", "visible"),
            "wood_pickaxe": self._pred("item", "wood_pickaxe", "positive"),
            "stone": self._pred("item", "stone", "positive"),
            "water": self._pred("terrain", "water", "visible"),
            "drink": self._pred("state", "drink", "restored"),
            "cow": self._pred("entity", "cow", "visible"),
            "food": self._pred("state", "food", "restored"),
            "lava": self._pred("terrain", "lava", "visible"),
            "health_risk": self._pred("state", "health", "risk"),
        }

        def norm(x):
            k = str(x).lower().strip().replace(" ", "_")

            if (
                k.startswith("collect_")
                or k.startswith("make_")
                or k.startswith("place_")
                or k.startswith("eat_")
                or k.startswith("drink_")
                or k.startswith("defeat_")
            ):
                return self._pred(
                    "achievement",
                    self.canonicalize_goal(k) or k,
                    "done",
                )

            return alias.get(k, x)

        u = norm(u)
        v = norm(v)

        if not u or not v or u == v:
            return None

        return (str(u), str(v), relation)

    def _normalize_goal(self, goal: Optional[str]) -> Optional[str]:
        if not goal:
            return None
        goal = str(goal).strip().lower().replace(" ", "_")
        if not goal:
            return None
        return goal

    def _state_afforded_goals(self, state: Dict[str, Any]) -> List[str]:
        visible = state["visible_objects"]
        goals: List[str] = []

        affordance_map = {
            "tree": "collect_wood",
            "water": "collect_drink",
            "stone": "collect_stone",
            "coal": "collect_coal",
            "iron": "collect_iron",
            "diamond": "collect_diamond",
            "cow": "eat_cow",
            "plant": "eat_plant",
            "zombie": "defeat_zombie",
            "skeleton": "defeat_skeleton",
        }

        for obj in visible:
            if obj in affordance_map:
                goals.append(affordance_map[obj])

        goals.append("survive")
        goals.append("explore")
        return goals

    def _extract_visible_objects(self, text_obs: str) -> List[str]:
        text = (text_obs or "").lower()
        vocab = [
            "tree", "stone", "coal", "iron", "diamond", "table", "furnace",
            "plant", "cow", "zombie", "skeleton", "water", "lava", "grass",
        ]
        return [token for token in vocab if token in text]

    def _parse_counts(self, text: str) -> Dict[str, int]:
        pairs = re.findall(r"(\d+)\s+([a-z_]+)", (text or "").lower())
        out: Dict[str, int] = {}
        for n, name in pairs:
            out[name] = int(n)
        return out

    def _map_collect_goal_to_visible_object(self, obj: str) -> str:
        mapping = {
            "wood": "tree",
            "sapling": "grass",
            "drink": "water",
        }
        return mapping.get(obj, obj)
    
    def build_skill_attempt_meta(
        self,
        goal: str,
        state: Dict[str, Any],
        belief_graph=None,
        candidate: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        effect = None
        if candidate is not None:
            effect = candidate.get("effect")
        if not effect:
            effect = self.goal_effect_predicate(goal, state)

        active = list(self.extract_active_predicates(state))

        prereqs = self.infer_goal_preconditions(
            goal=goal,
            state=state,
            belief_graph=belief_graph,
        ) if belief_graph is not None else []

        return {
            "goal": goal,
            "effect": effect,
            "start_predicates": active,
            "prerequisites": prereqs,
            "candidate_source": candidate.get("source") if isinstance(candidate, dict) else None,
            "frontier_confidence": float(candidate.get("frontier_confidence", 0.0)) if isinstance(candidate, dict) else 0.0,
        }