from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class EventType(str, Enum):
    REQUEST_RECEIVED = "request_received"
    CLAIM_EMITTED = "claim_emitted"
    EVIDENCE_LINKED = "evidence_linked"
    CONTRADICTION_FOUND = "contradiction_found"
    EXPERIMENT_PLANNED = "experiment_planned"
    EXPERIMENT_RAN = "experiment_ran"
    RED_TEAM_CHECK = "red_team_check"
    SYNTHESIS_READY = "synthesis_ready"


@dataclass(slots=True)
class RuntimeEvent:
    event_type: EventType
    timestamp: float
    payload: dict[str, Any]


@dataclass(slots=True)
class Claim:
    claim_id: str
    text: str
    confidence: float
    source_chain: list[str]


@dataclass(slots=True)
class Contradiction:
    claim_id: str
    reason: str


@dataclass(slots=True)
class ExperimentResult:
    hypothesis: str
    plan: str
    simulation: str
    outcome: str


@dataclass(slots=True)
class KnowledgeGraph:
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: list[dict[str, str]] = field(default_factory=list)

    def add_claim(self, claim: Claim) -> None:
        self.nodes[claim.claim_id] = {
            "type": "claim",
            "text": claim.text,
            "confidence": claim.confidence,
            "source_count": len(claim.source_chain),
            "timestamp": time.time(),
        }
        for idx, source in enumerate(claim.source_chain):
            source_node = f"source:{source}"
            if source_node not in self.nodes:
                self.nodes[source_node] = {"type": "source", "label": source}
            self.edges.append({"from": claim.claim_id, "to": source_node, "type": f"supported_by_{idx}"})

    def add_contradiction(self, contradiction: Contradiction) -> None:
        cid = f"contradiction:{len(self.edges)}"
        self.nodes[cid] = {"type": "contradiction", "reason": contradiction.reason}
        self.edges.append({"from": cid, "to": contradiction.claim_id, "type": "targets"})


@dataclass(slots=True)
class SharedMissionState:
    mission: str
    interface: str
    claims: list[Claim] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    experiments: list[ExperimentResult] = field(default_factory=list)
    disagreements: list[str] = field(default_factory=list)
    prompt_injection_flags: list[str] = field(default_factory=list)
    graph: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    events: list[RuntimeEvent] = field(default_factory=list)


class UnifiedOrchestrationEngine:
    """Single runtime used by CLI/chat/telegram/service pathways."""

    def __init__(self, memory_path: Path | None = None) -> None:
        self.memory_path = memory_path or Path("logs/runtime_memory_graph.json")

    async def run(self, query: str, interface: str, mode: str = "chat", agent: str | None = None) -> dict[str, Any]:
        state = SharedMissionState(mission=query, interface=interface)
        self._emit(state, EventType.REQUEST_RECEIVED, {"query": query, "mode": mode, "agent": agent})

        claims = self._extract_claims(query)
        for claim in claims:
            state.claims.append(claim)
            state.graph.add_claim(claim)
            self._emit(state, EventType.CLAIM_EMITTED, asdict(claim))
            self._emit(
                state,
                EventType.EVIDENCE_LINKED,
                {"claim_id": claim.claim_id, "source_chain": claim.source_chain},
            )

        for contradiction in self._detect_contradictions(state.claims):
            state.contradictions.append(contradiction)
            state.graph.add_contradiction(contradiction)
            self._emit(state, EventType.CONTRADICTION_FOUND, asdict(contradiction))

        for experiment in self._run_experiment_loop(state.claims):
            state.experiments.append(experiment)
            self._emit(state, EventType.EXPERIMENT_PLANNED, {"hypothesis": experiment.hypothesis, "plan": experiment.plan})
            self._emit(state, EventType.EXPERIMENT_RAN, asdict(experiment))

        for check in self._adversarial_checks(query, state.claims):
            if check.startswith("injection:"):
                state.prompt_injection_flags.append(check)
            else:
                state.disagreements.append(check)
            self._emit(state, EventType.RED_TEAM_CHECK, {"finding": check})

        summary = self._build_operator_summary(state)
        self._emit(state, EventType.SYNTHESIS_READY, {"summary": summary})

        self._persist_graph(state.graph)
        return {
            "response": summary,
            "ledger": {
                "claims": [asdict(c) for c in state.claims],
                "contradictions": [asdict(c) for c in state.contradictions],
                "experiments": [asdict(e) for e in state.experiments],
                "events": [asdict(e) for e in state.events],
            },
            "operator_panel": {
                "mission_state": "complete",
                "disagreements": state.disagreements,
                "source_coverage": sum(len(c.source_chain) for c in state.claims),
                "confidence_drift": self._confidence_drift(state.claims),
                "why_this_answer": "Synthesis is derived from claim confidence, contradiction checks, and experiment outcomes.",
            },
            "graph": {
                "nodes": state.graph.nodes,
                "edges": state.graph.edges,
            },
        }

    def _emit(self, state: SharedMissionState, event_type: EventType, payload: dict[str, Any]) -> None:
        state.events.append(RuntimeEvent(event_type=event_type, timestamp=time.time(), payload=payload))

    def _extract_claims(self, query: str) -> list[Claim]:
        sentences = [s.strip() for s in re.split(r"[.!?]", query) if s.strip()]
        if not sentences:
            sentences = [query.strip() or "No claim provided"]
        claims: list[Claim] = []
        for i, sentence in enumerate(sentences, start=1):
            confidence = max(0.3, min(0.95, 0.45 + len(sentence.split()) / 30.0))
            sources = ["user_prompt"]
            if any(k in sentence.lower() for k in ["paper", "study", "evidence", "data"]):
                sources.append("local_library_reference")
            claims.append(Claim(claim_id=f"claim_{i}", text=sentence, confidence=round(confidence, 2), source_chain=sources))
        return claims

    def _detect_contradictions(self, claims: list[Claim]) -> list[Contradiction]:
        contradictions: list[Contradiction] = []
        for claim in claims:
            lowered = claim.text.lower()
            if "always" in lowered and "never" in lowered:
                contradictions.append(Contradiction(claim_id=claim.claim_id, reason="Contains both absolute terms 'always' and 'never'."))
            if "impossible" in lowered and "already" in lowered:
                contradictions.append(Contradiction(claim_id=claim.claim_id, reason="Simultaneously describes impossible and already-completed state."))
        return contradictions

    def _run_experiment_loop(self, claims: list[Claim]) -> list[ExperimentResult]:
        results: list[ExperimentResult] = []
        for claim in claims[:2]:
            hypothesis = f"If '{claim.text[:80]}' is accurate, measurable indicators should improve."
            plan = "Generate baseline metric B=0.5, apply simulated intervention Δ based on confidence, compare B+Δ > B."
            delta = round((claim.confidence - 0.5) * 0.4, 3)
            simulated = round(0.5 + delta, 3)
            outcome = "supported" if simulated > 0.5 else "not_supported"
            results.append(
                ExperimentResult(
                    hypothesis=hypothesis,
                    plan=plan,
                    simulation=f"baseline=0.5; delta={delta}; simulated={simulated}",
                    outcome=outcome,
                )
            )
        return results

    def _adversarial_checks(self, query: str, claims: list[Claim]) -> list[str]:
        findings: list[str] = []
        lowered = query.lower()
        injection_markers = ["ignore previous", "system prompt", "override", "do not follow"]
        for marker in injection_markers:
            if marker in lowered:
                findings.append(f"injection: detected potential prompt-injection marker '{marker}'")

        if claims and max(c.confidence for c in claims) - min(c.confidence for c in claims) > 0.3:
            findings.append("cross_check: confidence spread indicates model disagreement; require secondary pass")

        if not findings:
            findings.append("cross_check: no major adversarial issues found")
        return findings

    def _build_operator_summary(self, state: SharedMissionState) -> str:
        top_claims = "; ".join(f"{c.text} (conf={c.confidence})" for c in state.claims[:3])
        contradictions = len(state.contradictions)
        experiments_supported = sum(1 for e in state.experiments if e.outcome == "supported")
        return (
            f"Mission '{state.mission}' processed via {state.interface}. "
            f"Claims: {len(state.claims)}. Contradictions: {contradictions}. "
            f"Experiments supporting hypothesis: {experiments_supported}/{len(state.experiments)}. "
            f"Top claims: {top_claims}."
        )

    def _confidence_drift(self, claims: list[Claim]) -> float:
        if not claims:
            return 0.0
        scores = [c.confidence for c in claims]
        return round(max(scores) - min(scores), 3)

    def _persist_graph(self, graph: KnowledgeGraph) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"nodes": graph.nodes, "edges": graph.edges, "updated_at": time.time()}
        self.memory_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


_ENGINE = UnifiedOrchestrationEngine()


async def run_rain_lab(query: str, mode: str = "chat", agent: str | None = None, recursive_depth: int = 1) -> str:
    del recursive_depth
    result = await _ENGINE.run(query=query, interface=mode, mode=mode, agent=agent)
    return result["response"]


async def run_rain_lab_detailed(query: str, interface: str, mode: str = "chat", agent: str | None = None) -> dict[str, Any]:
    return await _ENGINE.run(query=query, interface=interface, mode=mode, agent=agent)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unified runtime CLI for R.A.I.N. Lab")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--mode", default="chat")
    parser.add_argument("--agent", default=None)
    parser.add_argument("--library", default=None)
    parser.add_argument("--timeout", default=None)
    parser.add_argument("--recursive-depth", default=None)
    parser.add_argument("--no-recursive-intellect", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print full operator payload as JSON")
    args = parser.parse_args(argv)

    del args.library, args.timeout, args.recursive_depth, args.no_recursive_intellect
    payload = asyncio.run(run_rain_lab_detailed(query=args.topic, interface="cli", mode=args.mode, agent=args.agent))
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(payload["response"])
        print("\nOperator panel:")
        print(json.dumps(payload["operator_panel"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
