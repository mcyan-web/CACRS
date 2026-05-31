from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional


class BaseAdapter(ABC):
    """
    Unified adapter interface across environments.
    Each environment should implement how raw state is converted into a
    semantic state, how utility/cost are estimated, and how online evidence
    is extracted.
    """

    @abstractmethod
    def build_semantic_state(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def enumerate_goal_candidates(
        self,
        state: Dict[str, Any],
        llm_goals: List[str],
        mem_goals: List[str],
        belief_graph=None,
        skill_memory=None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def estimate_utility(
        self,
        goal: str,
        state: Dict[str, Any],
        skill_memory=None,
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def estimate_cost(
        self,
        goal: str,
        state: Dict[str, Any],
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def infer_support_path(
        self,
        goal: str,
        belief_graph,
        state: Dict[str, Any],
    ) -> List[Tuple[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def event_to_edge_evidence(
        self,
        prev_state: Dict[str, Any],
        action: Any,
        next_state: Dict[str, Any],
        goal: str,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def default_goal(self, state: Dict[str, Any]) -> str:
        raise NotImplementedError

    def extract_active_predicates(self, state: Dict[str, Any]) -> List[str]:
        return []

    def extract_task_objectives(self, state: Dict[str, Any], reward_info=None) -> List[str]:
        return []

    def extract_transition_events(self, prev_state: Dict[str, Any], action: Any, next_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []

    def predicate_to_goal(self, predicate: str) -> Optional[str]:
        return None

    def policy_input(self, obs, goal: str):
        """
        Optional: only really needed for MiniGrid if you want adapter-controlled
        policy input formatting.
        By default return obs unchanged.
        """
        return obs