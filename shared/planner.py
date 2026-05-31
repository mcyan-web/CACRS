from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math


@dataclass
class GoalScore:
    goal: str
    source: str
    success_prob: float
    support_confidence: float
    progress_signal: float
    value_gain: float
    expected_cost: float
    uncertainty: float
    skill_confidence: float = 0.0
    skill_dependency_satisfaction: float = 0.0
    risk: float = 0.0
    score: float = 0.0
    candidate: Any = None

class RiskAwarePlanner:
    """
    Submission-oriented planner.

    Core idea:
      score(goal | state)
        = log(expected_utility) + epistemic_bonus

        expected_utility
        = belief_feasibility
            * precondition_satisfaction
            * survival_probability
            * skill_reliability
            * (1 + task_utility)
            / sqrt(max(1, expected_cost))

        epistemic_bonus
        = skill_posterior_std * sqrt(belief_feasibility)

    Differences from old heuristic planner:
      - no hand-written utility / unlock linear combination
      - no direct task-specific curriculum logic in planner
      - success probability is estimated from:
          belief support confidence
          failure-memory historical success
          progress signal
      - uncertainty is estimated from:
          graph uncertainty
          insufficient experience
      - LLM query is event-triggered rather than fixed-interval
    """

    def __init__(
        self,
        beta_value_gain: float = 0.7,
        lambda_cost: float = 0.5,
        eta_uncertainty: float = 0.4,
        fail_streak_threshold: int = 20,
        progress_stall_threshold: float = 1e-3,
        min_attempts_for_low_sr: int = 20,
        low_success_rate_threshold: float = 0.02,
        low_support_conf_threshold: float = 0.35,
        cooldown_steps: int = 100,
    ):
        self.beta_value_gain = float(beta_value_gain)
        self.lambda_cost = float(lambda_cost)
        self.eta_uncertainty = float(eta_uncertainty)

        self.fail_streak_threshold = int(fail_streak_threshold)
        self.progress_stall_threshold = float(progress_stall_threshold)
        self.min_attempts_for_low_sr = int(min_attempts_for_low_sr)
        self.low_success_rate_threshold = float(low_success_rate_threshold)
        self.low_support_conf_threshold = float(low_support_conf_threshold)
        self.cooldown_steps = int(cooldown_steps)

    # ------------------------------------------------------------------
    # adaptive LLM gate
    # ------------------------------------------------------------------

    def should_query_llm(
        self,
        current_goal: Optional[str],
        state: Dict[str, Any],
        global_step: int,
        last_llm_query_step: int,
        belief_graph=None,
        failure_memory=None,
        adapter=None,
        candidates: Optional[List[Dict[str, Any]]] = None,
        frontier: Optional[List[Dict[str, Any]]] = None,
        min_candidates: int = 2,
        min_top_candidate_score: float = -1.0,
        min_graph_edges: int = 3,
    ) -> Tuple[bool, List[str]]:
        reasons: List[str] = []

        if global_step - last_llm_query_step < self.cooldown_steps:
            return False, reasons

        # 1. Cold-start: belief graph is empty or nearly empty.
        try:
            num_edges = len(getattr(belief_graph, "edges", {}) or {})
            if num_edges < int(min_graph_edges):
                reasons.append("belief_graph_cold_start")
        except Exception:
            pass

        # 2. No current goal.
        if current_goal is None or str(current_goal).strip() == "":
            reasons.append("no_current_goal")

        # 3. Candidate starvation.
        if candidates is not None:
            valid_candidates = [
                c for c in candidates
                if c is not None and str(c.get("goal", c.get("effect", ""))).strip()
            ]
            if len(valid_candidates) < int(min_candidates):
                reasons.append("candidate_starvation")

            # If planner has already attached scores, check quality.
            scored = []
            for c in valid_candidates:
                try:
                    if "score" in c:
                        scored.append(float(c["score"]))
                except Exception:
                    pass

            if scored and max(scored) < float(min_top_candidate_score):
                reasons.append("low_candidate_quality")

        # 4. Unresolved belief frontier.
        if frontier is not None:
            unresolved = []
            for f in frontier:
                source = str(f.get("source", ""))
                conf = float(f.get("path_confidence", 0.0))
                if source == "belief_target" or conf <= getattr(self, "low_support_conf_threshold", 0.35):
                    unresolved.append(f)

            if len(unresolved) > 0:
                reasons.append("unresolved_belief_frontier")

        # 5. Failure-memory based triggers.
        if current_goal is not None and failure_memory is not None:
            try:
                if int(failure_memory.fail_streak(current_goal)) >= self.fail_streak_threshold:
                    reasons.append("high_fail_streak")
            except Exception:
                pass

            try:
                attempts = int(failure_memory.num_attempts(current_goal))
                sr = float(failure_memory.recent_success_rate(current_goal))
                if attempts >= self.min_attempts_for_low_sr and sr < self.low_success_rate_threshold:
                    reasons.append("low_recent_success_rate")
            except Exception:
                pass

        # 6. Progress stall.
        if current_goal is not None and adapter is not None:
            try:
                progress_delta = adapter.goal_progress_delta(current_goal, state)
                if progress_delta is not None:
                    progress_delta = float(progress_delta)
                    if progress_delta <= self.progress_stall_threshold:
                        reasons.append("stalled_progress")
            except Exception as e:
                # Do not silently hide systematic interface mismatch.
                reasons.append("progress_delta_unavailable")

        # 7. Low belief support.
        if current_goal is not None:
            try:
                support_conf = self.support_confidence(
                    current_goal=current_goal,
                    state=state,
                    belief_graph=belief_graph,
                    adapter=adapter,
                )
                if support_conf < self.low_support_conf_threshold:
                    reasons.append("low_support_confidence")
            except Exception:
                reasons.append("support_confidence_unavailable")

        # 8. High graph uncertainty.
        try:
            if (
                    current_goal is not None
                    and frontier is not None
                    and belief_graph is not None
                    and hasattr(belief_graph, "edge_confidence")
                ):
                    relevant_nodes = set()

                    for f in frontier:
                        pred = f.get("predicate")
                        tgt = f.get("target")
                        if pred:
                            relevant_nodes.add(str(pred))
                        if tgt:
                            relevant_nodes.add(str(tgt))

                        for edge in f.get("support_path", []) or []:
                            try:
                                u, v = edge
                                relevant_nodes.add(str(u))
                                relevant_nodes.add(str(v))
                            except Exception:
                                pass

                    uncertain_scores = []
                    for (u, v), belief in getattr(belief_graph, "edges", {}).items():
                        if u not in relevant_nodes and v not in relevant_nodes:
                            continue

                        conf = float(belief.mean)
                        unc = 1.0 - abs(conf - 0.5) * 2.0
                        uncertain_scores.append(unc)

                    if uncertain_scores:
                        mean_unc = sum(uncertain_scores[:5]) / min(5, len(uncertain_scores))
                        if mean_unc >= 0.80:
                            reasons.append("high_relevant_belief_uncertainty")
        except Exception:
            pass

        # Deduplicate.
        deduped = []
        seen = set()
        for r in reasons:
            if r not in seen:
                seen.add(r)
                deduped.append(r)

        return len(deduped) > 0, deduped

    # ------------------------------------------------------------------
    # replan gate
    # ------------------------------------------------------------------

    def should_replan(
        self,
        current_goal: Optional[str],
        state: Dict[str, Any],
        belief_graph=None,
        failure_memory=None,
        adapter=None,
    ) -> Tuple[bool, List[str]]:
        reasons: List[str] = []

        if current_goal is None or str(current_goal).strip() == "":
            reasons.append("no_current_goal")
            return True, reasons

        try:
            if adapter is not None and adapter.goal_satisfied(current_goal, state):
                reasons.append("goal_satisfied")
                return True, reasons
        except Exception:
            pass

        if failure_memory is not None:
            try:
                if int(failure_memory.fail_streak(current_goal)) >= self.fail_streak_threshold:
                    reasons.append("high_fail_streak")
            except Exception:
                pass

            try:
                attempts = int(failure_memory.num_attempts(current_goal))
                sr = float(failure_memory.recent_success_rate(current_goal))
                if attempts >= self.min_attempts_for_low_sr and sr < self.low_success_rate_threshold:
                    reasons.append("low_recent_success_rate")
            except Exception:
                pass

        try:
            progress_delta = adapter.goal_progress_delta(current_goal, state)
            if progress_delta is not None:
                progress_delta = float(progress_delta)
                if progress_delta <= self.progress_stall_threshold:
                    reasons.append("stalled_progress")
        except Exception:
            pass

        return len(reasons) > 0, reasons

    # ------------------------------------------------------------------
    # goal selection
    # ------------------------------------------------------------------

    def select_goal(
        self,
        candidates: List[Any],
        state: Dict[str, Any],
        belief_graph=None,
        adapter=None,
        failure_memory=None,
        value_model=None,
        skill_memory=None,
    ) -> Tuple[str, Dict[str, Any]]:
        if adapter is None:
            raise ValueError("adapter must not be None")

        scored: List[GoalScore] = []

        for cand in candidates:
            goal, source = self._unpack_candidate(cand)
            if not goal:
                continue

            success_prob = self.estimate_success_probability(
                goal=goal,
                state=state,
                belief_graph=belief_graph,
                adapter=adapter,
                failure_memory=failure_memory,
            )
            support_conf = self.support_confidence(goal, state, belief_graph, adapter)
            if isinstance(cand, dict) and cand.get("frontier_confidence") is not None:
                try:
                    support_conf = max(float(support_conf), float(cand.get("frontier_confidence", 0.0)))
                except Exception:
                    pass
            progress_signal = self._safe_progress_signal(goal, state, adapter)
            value_gain = self._estimate_value_gain(goal, state, adapter, value_model)
            expected_cost = self._estimate_expected_cost(goal, state, adapter, failure_memory)
            uncertainty = self._estimate_uncertainty(goal, state, belief_graph, failure_memory, adapter)

            skill_features = {}
            if skill_memory is not None and hasattr(skill_memory, "planner_features"):
                try:
                    skill_features = skill_memory.planner_features(goal, candidate=cand if isinstance(cand, dict) else None, state=state, adapter=adapter)
                except Exception:
                    skill_features = {}
            skill_conf = float(skill_features.get("skill_confidence", 0.0))
            skill_dep = float(skill_features.get("skill_dependency_satisfaction", 0.0))
            skill_cost = float(skill_features.get("skill_expected_cost", 0.0))
            if skill_cost > 0.0:
                expected_cost = 0.5 * expected_cost + 0.5 * skill_cost

            risk = 0.0
            if belief_graph is not None and hasattr(belief_graph, "risk_score") and isinstance(cand, dict):
                try:
                    active = adapter.extract_active_predicates(state) if hasattr(adapter, "extract_active_predicates") else []
                    risk = float(belief_graph.risk_score(active, cand.get("effect") or goal))
                except Exception:
                    risk = 0.0

            skill_post_mean = float(skill_features.get("skill_posterior_mean", 0.5))
            skill_post_std = float(skill_features.get("skill_posterior_std", 0.25))
            skill_attempts = float(skill_features.get("skill_attempts", 0.0))

            # Belief feasibility is not just another weighted term.
            # It gates whether the skill is grounded in the entity graph.
            belief_feasibility = max(1e-4, min(1.0, support_conf))

            # Preconditions are also a gate.
            precond_sat = max(1e-4, min(1.0, skill_dep if skill_dep > 0.0 else progress_signal))

            # Risk is multiplicative survival probability, not merely a linear penalty.
            survival_prob = max(1e-4, 1.0 - min(1.0, risk))

            # Cost normalized as expected execution burden.
            cost = max(1.0, expected_cost)
            cost_factor = 1.0 / math.sqrt(cost)

            # Task utility.
            # If value model is unavailable, adapter.goal_novelty / terminal progress is used.
            task_utility = max(0.0, value_gain)

            # Bayesian reliability.
            reliability = skill_post_mean

            # Epistemic exploration bonus:
            # only useful when the belief graph says the skill is plausible.
            epistemic_bonus = skill_post_std * math.sqrt(belief_feasibility)

            # Final expected utility:
            # multiplicative gates reduce invalid skills without hand-written if-else curricula.
            expected_utility = (
                belief_feasibility
                * precond_sat
                * survival_prob
                * reliability
                * (1.0 + task_utility)
                * cost_factor
            )

            score = math.log(max(1e-8, expected_utility)) + epistemic_bonus

            scored.append(
                GoalScore(
                    goal=goal,
                    source=source,
                    success_prob=success_prob,
                    support_confidence=support_conf,
                    progress_signal=progress_signal,
                    value_gain=value_gain,
                    expected_cost=expected_cost,
                    uncertainty=uncertainty,
                    skill_confidence=skill_conf,
                    skill_dependency_satisfaction=skill_dep,
                    risk=risk,
                    score=score,
                    candidate=cand if isinstance(cand, dict) else {"goal": goal, "source": source},
                )
            )

        if len(scored) == 0:
            fallback_goal = adapter.default_goal(state)
            return fallback_goal, {
                "selected_goal": fallback_goal,
                "reason": "empty_candidates_fallback",
                "candidates": [],
                "score_margin": 0.0,
                "support_confidence": 0.3,
                "selected_source": "default",
            }

        scored.sort(key=lambda x: x.score, reverse=True)
        best = scored[0]
        margin = best.score - scored[1].score if len(scored) > 1 else float("inf")

        planner_info = {
            "selected_goal": best.goal,
            "reason": "belief_skill_risk_argmax",
            "selected_source": best.source,
            "score_margin": margin,
            "support_confidence": best.support_confidence,
            "success_prob": best.success_prob,
            "expected_cost": best.expected_cost,
            "uncertainty": best.uncertainty,
            "progress_signal": best.progress_signal,
            "candidates": [vars(x) for x in scored],
            "selected_candidate": best.candidate,
        }
        return best.goal, planner_info

    # ------------------------------------------------------------------
    # edge verification helper
    # ------------------------------------------------------------------

    def select_edges_for_verification(
        self,
        belief_graph,
        current_goals: List[str],
        max_edges: int = 5,
        adapter=None,
        states: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Tuple[str, str]]:
        """
        Pick most uncertain incoming support edges around current goals.
        """
        if belief_graph is None:
            return []

        proposed_edges: List[Tuple[str, str]] = []

        for i, goal in enumerate(current_goals):
            state = states[i] if (states is not None and i < len(states)) else {}
            try:
                path = adapter.infer_support_path(goal, state, belief_graph=belief_graph) if adapter is not None else []
            except Exception:
                path = []

            for e in path:
                if e not in proposed_edges:
                    proposed_edges.append(e)

        edge_with_uncertainty = []
        for e in proposed_edges:
            conf = self._safe_edge_confidence(belief_graph, e[0], e[1], default=0.3)
            uncertainty = 1.0 - abs(conf - 0.5) * 2.0
            edge_with_uncertainty.append((uncertainty, e))

        edge_with_uncertainty.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in edge_with_uncertainty[:max_edges]]

    # ------------------------------------------------------------------
    # probabilistic components
    # ------------------------------------------------------------------

    def estimate_success_probability(
        self,
        goal: str,
        state: Dict[str, Any],
        belief_graph=None,
        adapter=None,
        failure_memory=None,
    ) -> float:
        support_conf = self.support_confidence(goal, state, belief_graph, adapter)
        progress_signal = self._safe_progress_signal(goal, state, adapter)

        hist_sr = 0.5
        attempts = 0
        if failure_memory is not None:
            try:
                attempts = int(failure_memory.num_attempts(goal))
                if attempts > 0:
                    hist_sr = float(failure_memory.recent_success_rate(goal))
            except Exception:
                pass

        # multiplicative form is more robust than raw additive heuristic
        prob = max(1e-4, support_conf) * math.sqrt(max(1e-4, hist_sr)) * max(1e-4, 0.5 + 0.5 * progress_signal)

        # if no history at all, avoid overconfidence
        if attempts == 0:
            prob *= 0.85

        return min(1.0, max(1e-4, prob))

    def support_confidence(
        self,
        current_goal: str,
        state: Dict[str, Any],
        belief_graph=None,
        adapter=None,
    ) -> float:
        try:
            support_path = adapter.infer_support_path(current_goal, state, belief_graph=belief_graph)
        except Exception:
            support_path = []

        if len(support_path) == 0:
            return 0.3

        confs = [self._safe_edge_confidence(belief_graph, u, v, default=0.3) for (u, v) in support_path]
        return float(min(confs))

    def _estimate_value_gain(self, goal: str, state: Dict[str, Any], adapter=None, value_model=None) -> float:
        if adapter is None:
            return 0.0

        # optional learned hook
        if value_model is not None and hasattr(adapter, "counterfactual_value_gain"):
            try:
                return float(adapter.counterfactual_value_gain(goal, state, value_model))
            except Exception:
                pass

        try:
            return float(adapter.goal_novelty(goal, state))
        except Exception:
            return 0.0

    def _estimate_expected_cost(self, goal: str, state: Dict[str, Any], adapter=None, failure_memory=None) -> float:
        if failure_memory is not None:
            try:
                mean_steps = float(failure_memory.mean_steps_to_outcome(goal))
                if mean_steps > 0:
                    return mean_steps
            except Exception:
                pass

        try:
            return float(adapter.goal_distance_estimate(goal, state))
        except Exception:
            return 1.0

    def _estimate_uncertainty(self, goal: str, state: Dict[str, Any], belief_graph=None, failure_memory=None, adapter=None) -> float:
        support_conf = self.support_confidence(goal, state, belief_graph, adapter)
        u_graph = 1.0 - support_conf

        u_hist = 0.5
        if failure_memory is not None:
            try:
                attempts = int(failure_memory.num_attempts(goal))
                if attempts > 0:
                    u_hist = 1.0 / math.sqrt(1.0 + attempts)
            except Exception:
                pass

        return 0.5 * u_graph + 0.5 * u_hist

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _safe_progress_signal(self, goal: str, state: Dict[str, Any], adapter=None) -> float:
        try:
            return float(adapter.goal_progress_signal(goal, state))
        except Exception:
            return 0.0

    def _safe_edge_confidence(self, belief_graph, u: str, v: str, default: float = 0.3) -> float:
        if belief_graph is None:
            return default

        for name in ["edge_confidence", "get_edge_confidence", "confidence"]:
            fn = getattr(belief_graph, name, None)
            if callable(fn):
                try:
                    return float(fn(u, v))
                except Exception:
                    pass

        return default

    def _unpack_candidate(self, cand: Any) -> Tuple[Optional[str], str]:
        if isinstance(cand, dict):
            return cand.get("goal", None), cand.get("source", "unknown")
        if isinstance(cand, str):
            return cand, "unknown"
        return None, "unknown"