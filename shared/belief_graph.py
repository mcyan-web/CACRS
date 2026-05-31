from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional, Any
from collections import defaultdict


Edge = Tuple[str, str]


@dataclass
class EdgeBelief:
    alpha: float = 0.5
    beta: float = 1.5
    support_count: int = 0
    verified_count: int = 0
    last_updated_step: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def mean(self) -> float:
        denom = self.alpha + self.beta
        if denom <= 0:
            return 0.25
        return self.alpha / denom

    @property
    def variance(self) -> float:
        a = self.alpha
        b = self.beta
        denom = (a + b) ** 2 * (a + b + 1.0)
        if denom <= 0:
            return 0.0
        return (a * b) / denom

    def update_positive(self, weight: float = 1.0) -> None:
        self.alpha += max(0.0, float(weight))
        self.support_count += 1

    def update_negative(self, weight: float = 1.0) -> None:
        self.beta += max(0.0, float(weight))
        self.support_count += 1


class BayesianCausalBeliefGraph:
    """
    Entity-level Bayesian belief graph.

    Nodes:
        typed predicates, e.g.
        terrain:tree:visible()
        item:wood:positive()
        state:drink:restored()
        object:door:open(red)

    Edges:
        predicate -> predicate with relation type and Beta confidence.

    Supported edge proposal formats:
        (src, dst)
        (src, dst, relation)
        {"source": src, "target": dst, "relation": relation}
    """

    def __init__(
        self,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        conf_threshold: float = 0.35,
        topk_per_goal: int = 5,
        recency_window: int = 5000,
        default_missing_confidence: float = 0.3,
    ):
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)

        self.conf_threshold = float(conf_threshold)
        self.topk_per_goal = int(topk_per_goal)
        self.recency_window = int(recency_window)
        self.default_missing_confidence = float(default_missing_confidence)

        self.edges: Dict[Edge, EdgeBelief] = {}

    # ------------------------------------------------------------------
    # normalization
    # ------------------------------------------------------------------

    def _norm_node(self, x: Any) -> str:
        return str(x).strip()

    def _norm_relation(self, relation: Any) -> str:
        rel = str(relation or "enables").strip().lower()
        rel = rel.strip(" .;，。|")
        rel = rel.replace("-", "_")
        return rel or "enables"

    def _parse_edge_item(self, item: Any) -> Optional[Tuple[str, str, str, Optional[Dict[str, Any]]]]:
        """
        Return:
            (src, dst, relation, meta)
        """
        if item is None:
            return None

        if isinstance(item, dict):
            u = item.get("source") or item.get("src") or item.get("u")
            v = item.get("target") or item.get("dst") or item.get("v")
            relation = item.get("relation") or item.get("relation_type") or item.get("type") or "enables"
            meta = item.get("meta", None)

        elif isinstance(item, (list, tuple)):
            if len(item) < 2:
                return None
            u = item[0]
            v = item[1]
            relation = item[2] if len(item) >= 3 else "enables"
            meta = item[3] if len(item) >= 4 and isinstance(item[3], dict) else None

        else:
            return None

        u = self._norm_node(u)
        v = self._norm_node(v)
        relation = self._norm_relation(relation)

        if not u or not v or u == v:
            return None

        return u, v, relation, meta

    # ------------------------------------------------------------------
    # basic ops
    # ------------------------------------------------------------------

    def has_edge(self, u: str, v: str) -> bool:
        return (u, v) in self.edges

    def add_edge(
        self,
        u: str,
        v: str,
        relation: str = "enables",
        meta: Optional[Dict[str, Any]] = None,
        step: int = 0,
    ) -> None:
        u = self._norm_node(u)
        v = self._norm_node(v)
        relation = self._norm_relation(relation)

        if not u or not v or u == v:
            return

        edge = (u, v)
        edge_meta = dict(meta or {})
        edge_meta.setdefault("relation", relation)

        if edge not in self.edges:
            self.edges[edge] = EdgeBelief(
                alpha=self.prior_alpha,
                beta=self.prior_beta,
                last_updated_step=int(step),
                meta=edge_meta,
            )
        else:
            self.edges[edge].meta.update(edge_meta)
            self.edges[edge].last_updated_step = int(step)

    def ingest_edge_proposals(
        self,
        edge_candidates,
        step: int = 0,
        source: str = "llm_proposal",
        weight: float = 0.15,
        ) -> None:
        """
        Ingest LLM-proposed edges conservatively.

        LLM proposals should not be treated as strong positive evidence.
        They are hypotheses, not verified causal relations.
        """
        for item in edge_candidates or []:
            parsed = self._parse_edge_item(item)
            if parsed is None:
                continue

            u, v, relation, meta = parsed

            meta = dict(meta or {})
            meta["source"] = source
            meta["relation"] = self._norm_relation(relation)

            # Add edge but only weakly increase alpha.
            self.add_edge(
                u,
                v,
                relation=relation,
                meta=meta,
                step=step,
            )

            edge = (u, v)
            if edge in self.edges:
                self.edges[edge].alpha += float(weight)
                self.edges[edge].last_updated_step = int(step)

    # ------------------------------------------------------------------
    # confidence / uncertainty
    # ------------------------------------------------------------------

    def confidence(self, u: str, v: Optional[str] = None) -> float:
        """
        If called as confidence(u, v), return edge confidence.
        If called as confidence(node), return default missing confidence for compatibility.
        """
        if v is None:
            return self.default_missing_confidence

        belief = self.edges.get((u, v))
        if belief is None:
            return self.default_missing_confidence
        return belief.mean

    def edge_confidence(self, u: str, v: str) -> float:
        return self.confidence(u, v)

    def uncertainty(self, u: str, v: Optional[str] = None) -> float:
        return 1.0 - self.confidence(u, v)

    def edge_variance(self, u: str, v: str) -> float:
        belief = self.edges.get((u, v))
        if belief is None:
            return 0.25
        return belief.variance

    # ------------------------------------------------------------------
    # updates
    # ------------------------------------------------------------------

    def update_positive(
        self,
        u: str,
        v: str,
        weight: float = 1.0,
        step: int = 0,
        relation: str = "enables",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.add_edge(u, v, relation=relation, meta=meta, step=step)
        self.edges[(u, v)].update_positive(weight)
        self.edges[(u, v)].last_updated_step = int(step)

    def update_negative(
        self,
        u: str,
        v: str,
        weight: float = 1.0,
        step: int = 0,
        relation: str = "enables",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.add_edge(u, v, relation=relation, meta=meta, step=step)
        self.edges[(u, v)].update_negative(weight)
        self.edges[(u, v)].last_updated_step = int(step)

    def update_from_evidence(self, evidence_list: Iterable[Dict[str, Any]], step: int = 0) -> None:
        """
        Accepted item formats:
            {"edge": (u, v), "success": bool, "weight": 1.0, "meta": {...}}
            {"edge": (u, v, relation), "type": "positive"/"negative", "weight": 1.0}
            {"source": u, "target": v, "relation": relation, "success": bool}
        """
        for item in evidence_list or []:
            if not isinstance(item, dict):
                continue

            edge = item.get("edge", None)

            if edge is not None:
                parsed = self._parse_edge_item(edge)
            else:
                parsed = self._parse_edge_item(item)

            if parsed is None:
                continue

            u, v, relation_from_edge, meta_from_edge = parsed

            relation = item.get("relation") or item.get("relation_type") or relation_from_edge
            relation = self._norm_relation(relation)

            meta = dict(meta_from_edge or {})
            if isinstance(item.get("meta"), dict):
                meta.update(item["meta"])

            weight = float(item.get("weight", 1.0))

            if "success" in item:
                is_pos = bool(item["success"])
            else:
                is_pos = str(item.get("type", "positive")).lower() in {
                    "positive",
                    "pos",
                    "success",
                    "verified",
                    "true",
                }

            if is_pos:
                self.update_positive(u, v, weight=weight, step=step, relation=relation, meta=meta)
            else:
                self.update_negative(u, v, weight=weight, step=step, relation=relation, meta=meta)

    def update_from_verification(self, records: Iterable[Dict[str, Any]], step: int = 0) -> None:
        """
        Accepted item formats:
            {"edge": (u, v), "verified": bool, "weight": 1.0, "meta": {...}}
            {"edge": (u, v, relation), "success": bool, "weight": 1.0}
        """
        for rec in records or []:
            if not isinstance(rec, dict):
                continue

            edge = rec.get("edge", None)

            if edge is not None:
                parsed = self._parse_edge_item(edge)
            else:
                parsed = self._parse_edge_item(rec)

            if parsed is None:
                continue

            u, v, relation_from_edge, meta_from_edge = parsed

            relation = rec.get("relation") or rec.get("relation_type") or relation_from_edge
            relation = self._norm_relation(relation)

            meta = dict(meta_from_edge or {})
            if isinstance(rec.get("meta"), dict):
                meta.update(rec["meta"])

            weight = float(rec.get("weight", 1.0))

            if "verified" in rec:
                ok = bool(rec["verified"])
            else:
                ok = bool(rec.get("success", False))

            self.add_edge(u, v, relation=relation, meta=meta, step=step)

            if ok:
                self.edges[(u, v)].alpha += weight
            else:
                self.edges[(u, v)].beta += weight

            self.edges[(u, v)].verified_count += 1
            self.edges[(u, v)].last_updated_step = int(step)

    # ------------------------------------------------------------------
    # graph queries
    # ------------------------------------------------------------------

    def incoming_edges(self, goal: str) -> List[Edge]:
        return [e for e in self.edges.keys() if e[1] == goal]

    def outgoing_edges(self, source: str) -> List[Edge]:
        return [e for e in self.edges.keys() if e[0] == source]

    def top_uncertain_edges(self, k: int = 5) -> List[Tuple[Edge, float]]:
        scored = []
        for edge, belief in self.edges.items():
            uncertainty_score = 1.0 - abs(belief.mean - 0.5) * 2.0
            scored.append((edge, uncertainty_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def predecessors(
        self,
        target: str,
        min_conf: Optional[float] = None,
        relation_filter: Optional[Iterable[str]] = None,
    ) -> List[str]:
        threshold = self.conf_threshold if min_conf is None else float(min_conf)
        allowed = set(self._norm_relation(x) for x in (relation_filter or []))

        out: List[Tuple[float, str]] = []

        for (u, v), belief in self.edges.items():
            if v != target:
                continue

            rel = self._norm_relation(belief.meta.get("relation", "enables"))
            if allowed and rel not in allowed:
                continue

            if belief.mean >= threshold:
                out.append((belief.mean, u))

        out.sort(key=lambda x: x[0], reverse=True)
        return [u for _, u in out]

    def successors(
        self,
        source: str,
        min_conf: Optional[float] = None,
        relation_filter: Optional[Iterable[str]] = None,
    ) -> List[str]:
        threshold = self.conf_threshold if min_conf is None else float(min_conf)
        allowed = set(self._norm_relation(x) for x in (relation_filter or []))

        out: List[Tuple[float, str]] = []

        for (u, v), belief in self.edges.items():
            if u != source:
                continue

            rel = self._norm_relation(belief.meta.get("relation", "enables"))
            if allowed and rel not in allowed:
                continue

            if belief.mean >= threshold:
                out.append((belief.mean, v))

        out.sort(key=lambda x: x[0], reverse=True)
        return [v for _, v in out]

    def frontier_to_target(
        self,
        active_nodes: Iterable[str],
        target_nodes: Iterable[str],
        max_depth: int = 3,
        min_conf: float = 0.35,
        max_frontier: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Backward search from target predicates.

        Return unsatisfied frontier predicates on high-confidence paths to target.
        """
        active = set(active_nodes or [])
        targets = [t for t in (target_nodes or []) if t]

        frontier: Dict[str, Dict[str, Any]] = {}

        for target in targets:
            queue: List[Tuple[str, float, int, List[Tuple[str, str]]]] = [
                (target, 1.0, 0, [])
            ]
            visited = {target}

            while queue:
                node, path_conf, depth, path = queue.pop(0)

                if depth >= int(max_depth):
                    continue

                for src in self.predecessors(node, min_conf=min_conf):
                    edge_conf = self.edge_confidence(src, node)
                    new_conf = min(path_conf, edge_conf)
                    new_path = [(src, node)] + path

                    if src not in active:
                        prev = frontier.get(src)
                        if prev is None or new_conf > float(prev.get("path_confidence", 0.0)):
                            frontier[src] = {
                                "predicate": src,
                                "target": target,
                                "path_confidence": float(new_conf),
                                "depth": depth + 1,
                                "support_path": new_path,
                                "source": "belief_frontier",
                            }

                    if src not in visited:
                        visited.add(src)
                        queue.append((src, new_conf, depth + 1, new_path))

        # If graph cannot explain target yet, keep target itself as unresolved frontier.
        for target in targets:
            if target not in active and target not in frontier:
                frontier[target] = {
                    "predicate": target,
                    "target": target,
                    "path_confidence": self.default_missing_confidence,
                    "depth": 0,
                    "support_path": [],
                    "source": "belief_target",
                }

        arr = list(frontier.values())
        arr.sort(
            key=lambda x: (
                -float(x.get("path_confidence", 0.0)),
                int(x.get("depth", 0)),
            )
        )
        return arr[: int(max_frontier)]

    def risk_score(
        self,
        active_nodes: Iterable[str],
        candidate_effect: str,
        min_conf: float = 0.35,
    ) -> float:
        active = set(active_nodes or [])
        risk = 0.0

        for (u, v), belief in self.edges.items():
            if u not in active:
                continue

            rel = self._norm_relation(belief.meta.get("relation", "enables"))

            if rel in {"risks", "damages", "suppresses", "prevents"} and belief.mean >= min_conf:
                if v == candidate_effect or "risk" in v or "death" in v or "health" in v:
                    risk = max(risk, belief.mean)

        return float(risk)

    # ------------------------------------------------------------------
    # prompt export
    # ------------------------------------------------------------------

    def export_edges_for_prompt(
        self,
        max_edges: int = 20,
        min_confidence: Optional[float] = None,
    ) -> List[str]:
        threshold = self.conf_threshold if min_confidence is None else float(min_confidence)

        rows = []
        for (u, v), belief in self.edges.items():
            conf = float(belief.mean)
            if conf < threshold:
                continue

            relation = self._norm_relation(belief.meta.get("relation", "enables"))
            rows.append((conf, f"{u} -> {v} | {relation}."))

        rows.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in rows[: int(max_edges)]]
    def _edge_support_count(self, belief) -> float:
        """
        Number of non-prior observations/evidence units supporting this edge.

        Assumes Beta(alpha, beta) with prior_alpha / prior_beta.
        """
        try:
            return max(
                0.0,
                float(belief.alpha)
                + float(belief.beta)
                - float(self.prior_alpha)
                - float(self.prior_beta),
            )
        except Exception:
            return 0.0


    def _edge_source(self, belief) -> str:
        try:
            return str(belief.meta.get("source", belief.meta.get("level", "")))
        except Exception:
            return ""


    def _edge_priority_score(self, edge, belief, current_step: int) -> float:
        """
        Higher means more worth keeping.
        Combines confidence, verification, evidence support, and recency.
        """
        conf = float(belief.mean)
        support = self._edge_support_count(belief)
        verified = float(getattr(belief, "verified_count", 0))

        age = max(0, int(current_step) - int(getattr(belief, "last_updated_step", 0)))
        recency_bonus = max(0.0, 1.0 - age / max(1.0, float(self.recency_window)))

        source = self._edge_source(belief)

        # LLM-only edges should be easier to drop than env-supported edges.
        source_penalty = 0.0
        if source in {"llm", "llm_proposal", "language_model"}:
            source_penalty = 0.15

        # Verified/env-supported edges get priority.
        score = (
            conf
            + 0.08 * min(support, 5.0)
            + 0.15 * min(verified, 3.0)
            + 0.05 * recency_bonus
            - source_penalty
        )
        return float(score)
    # ------------------------------------------------------------------
    # pruning
    # ------------------------------------------------------------------

    def prune(
        self,
        current_step: int,
        *,
        verified_keep_conf: float = 0.55,
        evidence_keep_conf: float = 0.60,
        llm_keep_conf: float = 0.70,
        min_evidence_support: float = 2.0,
        recent_keep_steps: int = 1024,
        max_edges_total: int = 1200,
        topk_incoming_per_target: int = 8,
        topk_outgoing_per_source: int = 8,
        always_keep_verified: bool = True,
    ) -> None:
        """
        Conservative pruning for Bayesian belief graph.

        Main idea:
        - Do not keep weak LLM-only prior edges forever.
        - Keep verified or repeatedly environment-supported edges.
        - Temporarily keep recent edges so they can be verified.
        - Keep top-k incoming/outgoing edges to preserve useful graph structure.
        - Enforce a hard global edge budget.
        """
        if not self.edges:
            return

        keep = set()
        current_step = int(current_step)

        # ------------------------------------------------------------
        # 1) Evidence-aware keep rule
        # ------------------------------------------------------------
        for edge, belief in self.edges.items():
            conf = float(belief.mean)
            support = self._edge_support_count(belief)
            verified_count = int(getattr(belief, "verified_count", 0))
            age = current_step - int(getattr(belief, "last_updated_step", 0))
            source = self._edge_source(belief)

            is_llm_edge = source in {"llm", "llm_proposal", "language_model"}

            # A. Verified edges: keep if confidence is not too bad.
            if verified_count > 0:
                if always_keep_verified or conf >= verified_keep_conf:
                    keep.add(edge)
                    continue

            # B. Environment-supported edges: require enough support and higher confidence.
            if support >= min_evidence_support and conf >= evidence_keep_conf:
                keep.add(edge)
                continue

            # C. LLM-only edges: require much higher confidence.
            # This prevents Beta prior mean=0.5 edges from surviving forever.
            if is_llm_edge and conf >= llm_keep_conf:
                keep.add(edge)
                continue

            # D. Recent edges: keep only briefly, so they get a chance to collect evidence.
            if age <= int(recent_keep_steps):
                keep.add(edge)
                continue

        # ------------------------------------------------------------
        # 2) Keep top-k incoming edges per target
        # ------------------------------------------------------------
        incoming = {}
        for edge, belief in self.edges.items():
            _, target = edge
            incoming.setdefault(target, []).append(
                (self._edge_priority_score(edge, belief, current_step), edge)
            )

        for _, arr in incoming.items():
            arr.sort(key=lambda x: x[0], reverse=True)
            for _, edge in arr[: int(topk_incoming_per_target)]:
                keep.add(edge)

        # ------------------------------------------------------------
        # 3) Keep top-k outgoing edges per source
        # ------------------------------------------------------------
        outgoing = {}
        for edge, belief in self.edges.items():
            source, _ = edge
            outgoing.setdefault(source, []).append(
                (self._edge_priority_score(edge, belief, current_step), edge)
            )

        for _, arr in outgoing.items():
            arr.sort(key=lambda x: x[0], reverse=True)
            for _, edge in arr[: int(topk_outgoing_per_source)]:
                keep.add(edge)

        # ------------------------------------------------------------
        # 4) Hard global budget
        # ------------------------------------------------------------
        if len(keep) > int(max_edges_total):
            ranked = [
                (self._edge_priority_score(edge, self.edges[edge], current_step), edge)
                for edge in keep
            ]
            ranked.sort(key=lambda x: x[0], reverse=True)
            keep = set(edge for _, edge in ranked[: int(max_edges_total)])

        self.edges = {edge: self.edges[edge] for edge in keep}

    # ------------------------------------------------------------------
    # stats / compatibility
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        if not self.edges:
            return {
                "num_edges": 0,
                "mean_confidence": 0.0,
                "mean_variance": 0.0,
                "relation_counts": {},
            }

        means = [belief.mean for belief in self.edges.values()]
        variances = [belief.variance for belief in self.edges.values()]

        relation_counts: Dict[str, int] = {}
        for belief in self.edges.values():
            rel = self._norm_relation(belief.meta.get("relation", "enables"))
            relation_counts[rel] = relation_counts.get(rel, 0) + 1

        return {
            "num_edges": len(self.edges),
            "mean_confidence": sum(means) / len(means),
            "mean_variance": sum(variances) / len(variances),
            "relation_counts": relation_counts,
        }

    def all_nodes(self) -> List[str]:
        nodes = set()
        for u, v in self.edges.keys():
            nodes.add(u)
            nodes.add(v)
        return list(nodes)

    def edge_prob(self, u: str, v: str) -> float:
        """
        Compatibility alias for shaper/planner code.
        """
        return self.edge_confidence(u, v)

    def success_prob(self, u: str, v: Optional[str] = None) -> float:
        """
        Compatibility alias.
        - success_prob(u, v) -> edge confidence
        - success_prob(node) -> default missing confidence
        """
        if v is None:
            return self.default_missing_confidence
        return self.edge_confidence(u, v)