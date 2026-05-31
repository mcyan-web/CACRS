from __future__ import annotations

from typing import Dict, Any, List, Iterable, Tuple, Optional
import json
import re
_ALLOWED_RELATIONS = {
        "provides",
        "enables",
        "requires",
        "restores",
        "damages",
        "risks",
        "suppresses",
        "causes",
    }
class LLMProposer:
    """
    Lightweight wrapper around your existing language model interface.

    Responsibilities:
      1) propose causal edges
      2) propose candidate goals

    Non-responsibilities:
      - final goal selection
      - value estimation
      - low-level control
    """

    def __init__(
        self,
        lm,
        env_name: str,
        max_goal_candidates: int = 3,
    ):
        self.lm = lm
        self.env_name = env_name.lower()
        self.max_goal_candidates = int(max_goal_candidates)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propose_edges(self, semantic_state: Dict[str, Any]) -> List[Tuple[str, str]]:
        """
        Return a list of candidate causal edges.
        Output format:
            [("collect_wood", "place_table"), ("pickup_key", "open_door"), ...]
        """
        if self.env_name == "crafter":
            return self._propose_edges_crafter(semantic_state)
        if self.env_name == "minigrid":
            return self._propose_edges_minigrid(semantic_state)
        return []

    def propose_goals(self, semantic_state: Dict[str, Any], topk: Optional[int] = None) -> List[str]:
        """
        Return a ranked list of candidate goals.
        """
        topk = int(topk or self.max_goal_candidates)

        if self.env_name == "crafter":
            return self._propose_goals_crafter(semantic_state, topk=topk)
        if self.env_name == "minigrid":
            return self._propose_goals_minigrid(semantic_state, topk=topk)

        return []

    # ------------------------------------------------------------------
    # Crafter
    # ------------------------------------------------------------------

    def _propose_edges_crafter(self, state: Dict[str, Any]) -> List[Tuple[str, str]]:
        """
        Reuse your original Crafter LLM prompt interface as much as possible.

        Expected state keys:
            text_obs, inventory_text, status_text
        """

        text_obs = state.get("text_obs", "")
        inventory_text = state.get("inventory_text", "")
        status_text = state.get("status_text", "")
        prompt = {
            "task": "crafter_causal_edge_proposal",
            "instruction": (
                "Propose likely causal dependency edges between subgoals.\n"
                "Return one edge per line in the format: cause -> effect\n"
                "Do not explain."
            ),
            "examples": [
                {
                    "input": {
                        "text_obs": "You see a tree and a crafting table area.",
                        "inventory": "wood: 0",
                        "status": "energy: high"
                    },
                    "answer": "chop tree -> collect wood\ncollect wood -> place crafting table"
                },
                {
                    "input": {
                        "text_obs": "You see stone nearby.",
                        "inventory": "wood_pickaxe: 1",
                        "status": "energy: medium"
                    },
                    "answer": "make wood pickaxe -> mine stone"
                }
            ],
            "input": {
                "text_obs": text_obs,
                "inventory": inventory_text,
                "status": status_text,
            }
        }
        # If your original lm has a structured graph query API, use it here.
        # Example from Causal-aware-LLMs:
        #   str_graph = self.lm.query({'text_obs': text_obs, 'inv': inventory_text, 'status': status_text})
        #
        # Here we keep it generic.
        try:
            raw = self.lm.query(state)
            # print("raw edge proposal:", raw)
            edges = self._parse_edge_text(raw)
            # print("_parse_edge_text output:", edges)
            return edges
        except Exception:
            return []

    def _propose_goals_crafter(self, state: Dict[str, Any], topk: int) -> List[str]:
        text_obs = state.get("text_obs", state.get("obs", ""))
        inventory_text = state.get("inventory_text", state.get("inv", ""))
        status_text = state.get("status_text", state.get("status", ""))
        achievements = state.get("achievements", {})

        # try:
        state_dict = {
            "obs": text_obs,
            "status": status_text,
            "inv": inventory_text,
            "past_action": state.get("past_action", "<null>"),
            "past_goals": state.get("past_goals", "<null>"),
            "recent_changes": state.get("recent_changes", "<null>"),
            "recent_failures": state.get("recent_failures", "<null>"),
        }

        belief_edges = state.get("belief_edges", None)

        if hasattr(self.lm, "query_goal_for_complex_relation"):
            raw = self.lm.query_goal_for_complex_relation(
                verified_relation=belief_edges,
                state_dict=state,
                acheivement=achievements,
            )
        elif hasattr(self.lm, "query_goal"):
            raw = self.lm.query_goal(state)
        else:
            return []
        # print("LLM raw goal proposal:", raw)
        goals = self._parse_goal_list(raw)
        goals = [g for g in goals if isinstance(g, str) and len(g.strip()) > 0]
        return goals[:topk]

        # except Exception as e:
        #     print("[LLMProposer][Crafter goal proposal failed]", type(e).__name__, str(e))
        #     return []

    # ------------------------------------------------------------------
    # MiniGrid
    # ------------------------------------------------------------------

    def _propose_edges_minigrid(self, state: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Propose entity-level predicate relations for MiniGrid.

        The LLM is used only as a low-confidence prior proposer.  It must not
        output a final route or a task-specific curriculum.
        """
        mission = state.get("mission", "")
        visible_objects = state.get("visible_objects", [])
        carrying = state.get("carrying", "none")

        # try:
            # Keep semantic input compact.
        prompt = {
            "task": "minigrid_entity_relation_proposal",
            "mission": mission,
            "visible_objects": visible_objects,
            "carrying": carrying,
            "instruction": (
                "Propose direct entity/state causal relations as typed predicates. "
                "Return one relation per line in the form source -> target. "
                "Use predicates such as object:key:visible(red), "
                "state:agent:holding_key(red), object:door:visible(red), "
                "object:door:open(red), object:lava:visible(), "
                "state:agent:death_risk(). Use any when color is unknown. "
                "Do not output a final route, numbered steps, or explanations."
            ),
            "examples": [
                "object:key:visible(any) -> state:agent:holding_key(any)",
                "state:agent:holding_key(any) -> object:door:open(any)",
                "object:door:visible(any) -> object:door:open(any)",
                "object:lava:visible() -> state:agent:death_risk()",
                "state:agent:holding_key(red) -> object:door:open(red)"
            ],
        }

        if hasattr(self.lm, "query_edge_minigrid"):
            raw = self.lm.query_edge_minigrid(state)
            edges = self._parse_edge_text(raw)
            return edges
        # except Exception:
        #     pass

        return []

    def _propose_goals_minigrid(self, state: Dict[str, Any], topk: int) -> List[str]:
        mission = state.get("mission", "")
        visible_objects = state.get("visible_objects", [])
        carrying = state.get("carrying", "none")

        fallback = []

        # try:
        prompt = {
            "task": "minigrid_subgoal_proposal",
            "instruction": (
                f"Propose up to {topk} feasible next subgoals for solving the mission.\n"
                "Return one subgoal per line.\n"
                "Allowed style examples: explore, goto_key, pickup_key, goto_door, open_door.\n"
                "Do not explain."
            ),
            "examples": [
                {
                    "mission": "open the door",
                    "visible_objects": ["key"],
                    "carrying": "none",
                    "answer": "goto_key\npickup_key\nexplore"
                },
                {
                    "mission": "open the door and get to the goal",
                    "visible_objects": ["door"],
                    "carrying": "key",
                    "answer": "goto_door\nopen_door\nexplore"
                }
            ],
            "input": {
                "mission": mission,
                "visible_objects": visible_objects,
                "carrying": carrying,
            }
        }


        # raw = self.lm.query_goal_for_complex_relation_minigrid(state)
        if self.lm is None:
            return []
        if hasattr(self.lm, "query_goal_for_complex_relation_minigrid"):
            raw = self.lm.query_goal_for_complex_relation_minigrid(state)
        elif hasattr(self.lm, "query_goal"):
            raw = self.lm.query_goal(state)
        else:
            return []
        goals = self._parse_goal_list(raw)
        # print("LLM raw goal proposal:", raw)
        # print("LLM proposed goals:", goals)
        # except Exception:
        #     goals = []

        out = []
        seen = set()
        for g in goals + fallback:
            if g not in seen:
                seen.add(g)
                out.append(g)
        return out[:topk]

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    


    def _extract_llm_text(self, raw: Any) -> Any:
        """
        Convert common LLM response objects into plain text when possible.
        If raw is already list/dict, keep it for structured parsing.
        """
        if raw is None:
            return None

        if isinstance(raw, (str, list, dict)):
            return raw

        # OpenAI ChatCompletion-like object: raw.choices[0].message.content
        try:
            choices = getattr(raw, "choices", None)
            if choices:
                msg = getattr(choices[0], "message", None)
                if msg is not None:
                    content = getattr(msg, "content", None)
                    if content is not None:
                        return content

                    if isinstance(msg, dict) and "content" in msg:
                        return msg["content"]

                text = getattr(choices[0], "text", None)
                if text is not None:
                    return text
        except Exception:
            pass

        # ChatCompletionMessage-like object: raw.content
        try:
            content = getattr(raw, "content", None)
            if content is not None:
                return content
        except Exception:
            pass

        # Some SDKs return object with output_text.
        try:
            output_text = getattr(raw, "output_text", None)
            if output_text is not None:
                return output_text
        except Exception:
            pass

        # Last resort: stringify only if it looks like it contains useful text.
        s = str(raw)
        if "->" in s or "relations" in s:
            return s

        return raw


    def _clean_llm_line(self, text: str) -> str:
        """
        Remove markdown/list markers without damaging predicate syntax.
        """
        text = str(text).strip()

        # Remove code fence lines.
        if text.startswith("```"):
            return ""

        # Remove markdown bullet or numbered list prefix.
        text = re.sub(r"^\s*[-*]\s*", "", text)
        text = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", text)

        return text.strip()


    def _normalize_predicate_text(self, text: Any) -> str:
        """
        Preserve predicate syntax:
        entity:cow:visible()
        item:wood:positive()
        state:food:restored()
        """
        text = str(text).strip()
        text = text.strip("` \t\r\n")
        text = text.rstrip(".;，。")
        text = re.sub(r"\s+", "", text)
        return text


    def _normalize_relation_type(self, relation: Any) -> str:
        relation = str(relation or "enables").strip().lower()
        relation = relation.strip("` .;，。|")
        relation = relation.replace("-", "_")

        if relation in _ALLOWED_RELATIONS:
            return relation

        return "enables"


    def _canonicalize_predicate_for_belief_graph(self, pred: str) -> str:
        """
        Schema-level canonicalization. This is not task-order heuristic;
        it only fixes invalid predicate namespaces from LLM output.
        """
        pred = self._normalize_predicate_text(pred)

        # Common LLM mistake:
        # state:wood:available() should be item:wood:positive()
        m = re.fullmatch(r"state:([a-zA-Z0-9_]+):available\(\)", pred)
        if m:
            name = m.group(1)
            resource_like = {
                "wood",
                "stone",
                "coal",
                "iron",
                "diamond",
                "sapling",
                "wood_pickaxe",
                "stone_pickaxe",
                "iron_pickaxe",
                "wood_sword",
                "stone_sword",
                "iron_sword",
            }
            if name in resource_like:
                return f"item:{name}:positive()"

        return pred


    def _valid_predicate_text(self, pred: str) -> bool:
        """
        Basic predicate syntax validation.
        """
        if not pred:
            return False

        # domain:name:attr(args)
        return bool(
            re.fullmatch(
                r"(terrain|entity|object|item|state|achievement):[a-zA-Z0-9_]+:[a-zA-Z0-9_]+\([^)]*\)",
                pred,
            )
        )


    def _parse_edge_text(self, raw: Any) -> List[Tuple[str, str, str]]:
        """
        Parse LLM relation outputs into belief-graph edges.

        Supported:
        -  entity:cow:visible() -> state:food:restored() | restores.
        -  terrain:tree:visible() -> item:wood:positive() | provides.
        {"relations": [{"source": "...", "target": "...", "relation": "provides"}]}
        [("src", "dst")]
        [("src", "dst", "relation")]
        [{"source": "src", "target": "dst", "relation": "provides"}]
        """
        # print("[LLM RAW TYPE]", type(raw))
        # print("[LLM RAW REPR]", repr(raw)[:1000])
        raw = self._extract_llm_text(raw)

        if raw is None:
            return []

        if isinstance(raw, list):
            edges: List[Tuple[str, str, str]] = []

            for item in raw:
                if isinstance(item, dict):
                    src = item.get("source") or item.get("src") or item.get("u")
                    dst = item.get("target") or item.get("dst") or item.get("v")
                    rel = item.get("relation") or item.get("relation_type") or item.get("type") or "enables"

                    if src and dst:
                        src = self._canonicalize_predicate_for_belief_graph(src)
                        dst = self._canonicalize_predicate_for_belief_graph(dst)
                        rel = self._normalize_relation_type(rel)

                        if src != dst and self._valid_predicate_text(src) and self._valid_predicate_text(dst):
                            edges.append((src, dst, rel))

                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    src = self._canonicalize_predicate_for_belief_graph(item[0])
                    dst = self._canonicalize_predicate_for_belief_graph(item[1])
                    rel = self._normalize_relation_type(item[2] if len(item) >= 3 else "enables")

                    if src != dst and self._valid_predicate_text(src) and self._valid_predicate_text(dst):
                        edges.append((src, dst, rel))

            return edges

        if isinstance(raw, dict):
            if "relations" in raw and isinstance(raw["relations"], list):
                return self._parse_edge_text(raw["relations"])

            # Some APIs return {"content": "..."}.
            if "content" in raw:
                return self._parse_edge_text(raw["content"])

            # Some APIs return {"choices": [{"message": {"content": "..."}}]}.
            if "choices" in raw:
                try:
                    return self._parse_edge_text(raw["choices"][0]["message"]["content"])
                except Exception:
                    pass

            src = raw.get("source") or raw.get("src") or raw.get("u")
            dst = raw.get("target") or raw.get("dst") or raw.get("v")
            rel = raw.get("relation") or raw.get("relation_type") or raw.get("type") or "enables"

            if src and dst:
                src = self._canonicalize_predicate_for_belief_graph(src)
                dst = self._canonicalize_predicate_for_belief_graph(dst)
                rel = self._normalize_relation_type(rel)

                if src != dst and self._valid_predicate_text(src) and self._valid_predicate_text(dst):
                    return [(src, dst, rel)]

            return []

        if isinstance(raw, str):
            text = raw.strip()

            if not text or text.upper() == "NULL":
                return []

            # Remove code fences but keep inner lines.
            text = text.replace("```text", "").replace("```python", "").replace("```", "").strip()

            if text.startswith("{") or text.startswith("["):
                try:
                    parsed = json.loads(text)
                    return self._parse_edge_text(parsed)
                except Exception:
                    pass

            edges: List[Tuple[str, str, str]] = []

            for line in text.splitlines():
                line = self._clean_llm_line(line)

                if not line or line.upper() == "NULL":
                    continue

                if "->" not in line:
                    continue

                left, right = line.split("->", 1)

                if "|" in right:
                    dst_text, rel_text = right.split("|", 1)
                else:
                    dst_text, rel_text = right, "enables"

                src = self._canonicalize_predicate_for_belief_graph(left)
                dst = self._canonicalize_predicate_for_belief_graph(dst_text)
                rel = self._normalize_relation_type(rel_text)

                if not src or not dst:
                    continue

                if src == dst:
                    continue

                if not self._valid_predicate_text(src):
                    continue

                if not self._valid_predicate_text(dst):
                    continue

                edges.append((src, dst, rel))

            return edges

        return []

    def _parse_goal_list(self, raw: Any) -> List[str]:
        """
        Parse LLM-proposed subgoals as effect predicates.

        Supported examples:
        item:wood:positive()
        object:table:available()
        state:drink:restored()

        Also supports JSON:
        {"candidate_effects": [{"effect": "item:wood:positive()", "confidence": 0.8}]}
        ["item:wood:positive()", "state:drink:restored()"]

        Returns:
            List[str]: normalized effect predicates.
        """
        if raw is None:
            return []

        if isinstance(raw, list):
            goals: List[str] = []

            for item in raw:
                if isinstance(item, dict):
                    effect = (
                        item.get("effect")
                        or item.get("goal")
                        or item.get("predicate")
                        or item.get("target")
                    )
                    if effect:
                        effect = self._normalize_predicate_text(effect)
                        if effect and effect.upper() != "NULL":
                            goals.append(effect)

                else:
                    effect = self._normalize_predicate_text(item)
                    if effect and effect.upper() != "NULL":
                        goals.append(effect)

            return self._deduplicate_preserve_order(goals)

        if isinstance(raw, dict):
            if "candidate_effects" in raw and isinstance(raw["candidate_effects"], list):
                return self._parse_goal_list(raw["candidate_effects"])

            if "goals" in raw and isinstance(raw["goals"], list):
                return self._parse_goal_list(raw["goals"])

            effect = (
                raw.get("effect")
                or raw.get("goal")
                or raw.get("predicate")
                or raw.get("target")
            )

            if effect:
                effect = self._normalize_predicate_text(effect)
                return [effect] if effect and effect.upper() != "NULL" else []

            return []

        if isinstance(raw, str):
            text = raw.strip()

            if not text or text.upper() == "NULL":
                return []

            # JSON string output.
            if text.startswith("{") or text.startswith("["):
                try:
                    parsed = json.loads(text)
                    return self._parse_goal_list(parsed)
                except Exception:
                    pass

            chunks: List[str] = []

            for line in text.splitlines():
                line = self._clean_llm_line(line)

                if not line or line.upper() == "NULL":
                    continue

                # Support comma-separated output, but do not split inside predicate args.
                # Most current prompts ask for one predicate per line.
                if "," in line and not re.search(r"\([^)]*,[^)]*\)", line):
                    chunks.extend([x.strip() for x in line.split(",") if x.strip()])
                else:
                    chunks.append(line)

            cleaned: List[str] = []

            for chunk in chunks:
                effect = self._normalize_predicate_text(chunk)

                # If LLM accidentally outputs "goal: xxx", keep only the right side.
                for prefix in ("goal:", "effect:", "predicate:", "subgoal:"):
                    if effect.lower().startswith(prefix):
                        effect = effect[len(prefix):].strip()

                effect = self._normalize_predicate_text(effect)

                if effect and effect.upper() != "NULL":
                    cleaned.append(effect)

            return self._deduplicate_preserve_order(cleaned)

        return []


    def _deduplicate_preserve_order(self, items: List[str]) -> List[str]:
        seen = set()
        out = []

        for item in items:
            if item not in seen:
                seen.add(item)
                out.append(item)

        return out