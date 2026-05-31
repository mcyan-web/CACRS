from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass(frozen=True)
class Predicate:
    """Typed predicate used by the shared entity-level belief graph.

    The string format is intentionally compact and stable so existing PPO
    goal-conditioning code can still consume it after adapter binding.
    """

    domain: str
    name: str
    attr: str
    args: Tuple[Any, ...] = ()

    def key(self) -> str:
        args = ",".join(str(x) for x in self.args)
        return f"{self.domain}:{self.name}:{self.attr}({args})"


def pred(domain: str, name: str, attr: str, *args: Any) -> str:
    return Predicate(domain=domain, name=name, attr=attr, args=tuple(args)).key()


def is_predicate_key(x: str) -> bool:
    return isinstance(x, str) and x.count(":") >= 2 and x.endswith(")")
