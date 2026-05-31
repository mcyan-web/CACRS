from __future__ import annotations

from collections import deque, defaultdict
from typing import Dict, Any, List, Optional, Tuple


class AttemptBasedFailureMemory:
    """
    Attempt-based failure memory for replanning.

    Key idea:
      - A "goal attempt" starts when a goal becomes active for an env.
      - It ends when one of the following happens:
          1) the goal is satisfied
          2) replanning switches away from that goal
          3) the episode ends
          4) optional timeout is reached
      - fail_streak is updated ONLY when an attempt ends.

    This fixes the pathological behavior of step-based failure counting,
    where long-horizon goals accumulate enormous fail streaks simply
    because they are not completed at every timestep.
    """

    def __init__(self, history_size: int = 50):
        self.history_size = int(history_size)

        # finalized attempt history per goal
        self._history: Dict[str, deque] = {}

        # consecutive failed attempts per goal
        self._fail_streak: Dict[str, int] = defaultdict(int)

        # active attempts per env
        # env_index -> attempt dict
        self._active_attempts: Dict[int, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start_attempt(
        self,
        env_index: int,
        goal: str,
        start_step: int,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Start a new active attempt for (env_index, goal).
        If the same goal is already active for this env, do nothing.
        If another goal is active, caller should finalize it first.
        """
        if not goal:
            return

        current = self._active_attempts.get(int(env_index), None)
        if current is not None and current.get("goal") == goal:
            return

        self._active_attempts[int(env_index)] = {
            "goal": str(goal),
            "start_step": int(start_step),
            "steps": 0,
            "meta": meta or {},
        }

    def step_attempt(
        self,
        env_index: int,
        step_inc: int = 1,
    ) -> None:
        """
        Increment the duration of the currently active attempt for this env.
        Call once per environment step after the step is taken.
        """
        cur = self._active_attempts.get(int(env_index), None)
        if cur is None:
            return
        cur["steps"] = int(cur.get("steps", 0)) + int(step_inc)

    def end_attempt(
        self,
        env_index: int,
        success: bool,
        end_step: Optional[int] = None,
        reason: str = "terminated",
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Finalize the active attempt for env_index.
        Updates history and fail_streak.
        """
        env_index = int(env_index)
        cur = self._active_attempts.get(env_index, None)
        if cur is None:
            return None

        goal = cur["goal"]
        if goal not in self._history:
            self._history[goal] = deque(maxlen=self.history_size)

        record = {
            "goal": goal,
            "success": bool(success),
            "steps": int(cur.get("steps", 0)),
            "start_step": int(cur.get("start_step", -1)),
            "end_step": int(end_step) if end_step is not None else None,
            "env_index": env_index,
            "reason": str(reason),
            "meta": self._merge_meta(cur.get("meta", {}), meta or {}),
        }
        self._history[goal].append(record)

        if bool(success):
            self._fail_streak[goal] = 0
        else:
            self._fail_streak[goal] = int(self._fail_streak.get(goal, 0)) + 1

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
        """
        Convenience helper:
          finalize current active attempt (if any),
          then start a new attempt for new_goal.
        """
        old_record = None
        cur = self._active_attempts.get(int(env_index), None)

        if cur is not None:
            cur_goal = cur.get("goal")
            if cur_goal != new_goal:
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
        return old_record, self._active_attempts.get(int(env_index), None)

    def finalize_env(
        self,
        env_index: int,
        success: bool = False,
        end_step: Optional[int] = None,
        reason: str = "episode_end",
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Finalize the currently active attempt for one env.
        Useful at episode termination or forced reset.
        """
        return self.end_attempt(
            env_index=env_index,
            success=success,
            end_step=end_step,
            reason=reason,
            meta=meta,
        )

    def finalize_all(
        self,
        success: bool = False,
        end_step: Optional[int] = None,
        reason: str = "finalize_all",
        meta: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        out = []
        for env_index in list(self._active_attempts.keys()):
            rec = self.end_attempt(
                env_index=env_index,
                success=success,
                end_step=end_step,
                reason=reason,
                meta=meta,
            )
            if rec is not None:
                out.append(rec)
        return out

    # ------------------------------------------------------------------
    # timeout helper
    # ------------------------------------------------------------------

    def maybe_timeout(
        self,
        env_index: int,
        max_attempt_steps: int,
        step_idx: int,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        If the active attempt exceeds max_attempt_steps, finalize it as failure.
        """
        cur = self._active_attempts.get(int(env_index), None)
        if cur is None:
            return None

        if int(cur.get("steps", 0)) >= int(max_attempt_steps):
            return self.end_attempt(
                env_index=env_index,
                success=False,
                end_step=step_idx,
                reason="attempt_timeout",
                meta=meta,
            )
        return None

    # ------------------------------------------------------------------
    # queries used by planner
    # ------------------------------------------------------------------

    def fail_streak(self, goal: str) -> int:
        return int(self._fail_streak.get(goal, 0))

    def recent_success_rate(self, goal: str) -> float:
        hist = self._history.get(goal, None)
        if hist is None or len(hist) == 0:
            return 0.0
        succ = sum(1 for x in hist if bool(x["success"]))
        return succ / len(hist)

    def num_attempts(self, goal: str) -> int:
        hist = self._history.get(goal, None)
        return 0 if hist is None else len(hist)

    def mean_steps_to_outcome(self, goal: str) -> float:
        hist = self._history.get(goal, None)
        if hist is None or len(hist) == 0:
            return 0.0
        vals = [max(1, int(x["steps"])) for x in hist]
        return sum(vals) / len(vals)

    def recent_records(self, goal: str) -> List[Dict[str, Any]]:
        hist = self._history.get(goal, None)
        if hist is None:
            return []
        return list(hist)

    def active_goal(self, env_index: int) -> Optional[str]:
        cur = self._active_attempts.get(int(env_index), None)
        if cur is None:
            return None
        return cur.get("goal")

    def active_steps(self, env_index: int) -> int:
        cur = self._active_attempts.get(int(env_index), None)
        if cur is None:
            return 0
        return int(cur.get("steps", 0))

    def has_active_attempt(self, env_index: int) -> bool:
        return int(env_index) in self._active_attempts

    # ------------------------------------------------------------------
    # stats / reset
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        goals = list(self._history.keys())
        if not goals:
            return {
                "num_goals": 0,
                "num_active_attempts": len(self._active_attempts),
                "mean_fail_streak": 0.0,
                "mean_recent_success_rate": 0.0,
                "mean_attempt_length": 0.0,
            }

        mean_fail = sum(self.fail_streak(g) for g in goals) / len(goals)
        mean_sr = sum(self.recent_success_rate(g) for g in goals) / len(goals)
        mean_len = sum(self.mean_steps_to_outcome(g) for g in goals) / len(goals)

        return {
            "num_goals": len(goals),
            "num_active_attempts": len(self._active_attempts),
            "mean_fail_streak": mean_fail,
            "mean_recent_success_rate": mean_sr,
            "mean_attempt_length": mean_len,
        }

    def reset_goal(self, goal: str) -> None:
        self._fail_streak[goal] = 0
        if goal in self._history:
            self._history[goal].clear()

        # clear any active attempts on this goal
        to_remove = []
        for env_i, attempt in self._active_attempts.items():
            if attempt.get("goal") == goal:
                to_remove.append(env_i)
        for env_i in to_remove:
            del self._active_attempts[env_i]

    def reset_env(self, env_index: int) -> None:
        env_index = int(env_index)
        if env_index in self._active_attempts:
            del self._active_attempts[env_index]

    def reset_all(self) -> None:
        self._fail_streak.clear()
        self._history.clear()
        self._active_attempts.clear()

    # ------------------------------------------------------------------
    # backward-compatible wrapper
    # ------------------------------------------------------------------

    def update(
        self,
        goal: str,
        success: bool,
        steps: int = 1,
        env_index: int = 0,
        meta: Dict[str, Any] | None = None,
    ) -> None:
        """
        Backward-compatibility wrapper.

        IMPORTANT:
          This no longer means "one step failure update".
          It means: finalize one attempt record immediately.

        Keep this only so old code does not crash; for correct behavior,
        use start_attempt / step_attempt / end_attempt instead.
        """
        if not goal:
            return

        # if there is no active attempt, create a short synthetic one
        if not self.has_active_attempt(env_index):
            self.start_attempt(env_index=env_index, goal=goal, start_step=-1, meta=meta)
            # manually set steps
            cur = self._active_attempts.get(int(env_index), None)
            if cur is not None:
                cur["steps"] = int(steps)

        self.end_attempt(
            env_index=env_index,
            success=bool(success),
            end_step=None,
            reason="legacy_update",
            meta=meta,
        )

    # ------------------------------------------------------------------
    # utils
    # ------------------------------------------------------------------

    def _merge_meta(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        out.update(a or {})
        out.update(b or {})
        return out


# optional alias for drop-in replacement in older imports
SimpleFailureMemory = AttemptBasedFailureMemory