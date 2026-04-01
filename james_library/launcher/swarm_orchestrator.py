"""Peer Review Swarm Simulation for R.A.I.N. Lab.

Spins up temporary adversarial reviewer agents to debate a research document,
then synthesizes their critiques into a structured Peer_Review_Report.md.

Fully async and isolated from the main R.A.I.N. Lab agent context.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from james_library.utilities import context_manager, prefetch
from james_library.utilities.cost_monitor import BudgetExceededError, CostMonitor

logger = logging.getLogger(__name__)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback
    tomllib = None


BudgetPromptHandler = Callable[[BudgetExceededError, float], float | None]


# ---------------------------------------------------------------------------
# 1. Reviewer Persona Generator
# ---------------------------------------------------------------------------

# Maps broad topic keywords to specialized reviewer archetypes.
_DOMAIN_PERSONAS: dict[str, list[dict[str, str]]] = {
    "physics": [
        {
            "role": "Skeptical Physicist",
            "focus": "conservation laws, dimensional analysis, thermodynamic limits",
            "attack": "demand numerical estimates and unit-checked derivations for every claim",
        },
        {
            "role": "Rigorous Mathematician",
            "focus": "proof structure, convergence, boundary conditions, analytic continuation",
            "attack": "reject hand-waving; require explicit axioms, lemmas, and QED closures",
        },
        {
            "role": "Experimentalist",
            "focus": "measurement methodology, error bars, reproducibility, control experiments",
            "attack": "challenge any prediction that lacks a concrete, falsifiable experimental protocol",
        },
        {
            "role": "Adversarial Statistician",
            "focus": "p-hacking, overfitting, sample size, Bayesian priors, multiple comparisons",
            "attack": "flag every statistical claim that lacks power analysis or confidence intervals",
        },
    ],
    "biology": [
        {
            "role": "Molecular Biologist",
            "focus": "pathway specificity, off-target effects, protein-protein interactions",
            "attack": "demand Western blots, knockouts, or CRISPR controls for mechanistic claims",
        },
        {
            "role": "Biostatistician",
            "focus": "sample sizes, multiple testing correction, effect sizes, replication",
            "attack": "reject any conclusion drawn from n < 3 or uncorrected p-values",
        },
        {
            "role": "Evolutionary Skeptic",
            "focus": "selection pressure, phylogenetic confounds, neutral drift",
            "attack": "challenge adaptationist narratives that lack phylogenetic comparative analysis",
        },
    ],
    "computer_science": [
        {
            "role": "Complexity Theorist",
            "focus": "asymptotic bounds, NP-hardness reductions, approximation ratios",
            "attack": "demand formal runtime proofs; reject empirical-only complexity claims",
        },
        {
            "role": "Systems Adversary",
            "focus": "concurrency bugs, cache coherence, failure modes, tail latency",
            "attack": "stress-test every architecture claim with adversarial workload scenarios",
        },
        {
            "role": "Formal Methods Purist",
            "focus": "invariants, pre/post-conditions, model checking, type safety",
            "attack": "reject correctness claims without formal specification or proof sketch",
        },
    ],
    "default": [
        {
            "role": "Methodological Skeptic",
            "focus": "internal validity, confounds, causal inference, logical fallacies",
            "attack": "systematically enumerate every hidden assumption and logical gap",
        },
        {
            "role": "Quantitative Auditor",
            "focus": "numerical accuracy, unit consistency, order-of-magnitude sanity checks",
            "attack": "verify every number; flag unsourced statistics and suspiciously round figures",
        },
        {
            "role": "Reproducibility Enforcer",
            "focus": "data availability, code sharing, protocol detail, independent replication",
            "attack": "reject any result that cannot be independently reproduced from the paper alone",
        },
        {
            "role": "Logical Rigorist",
            "focus": "deductive validity, modus ponens chains, hidden premises, circular reasoning",
            "attack": "map the full argument graph and flag every non-sequitur or unstated axiom",
        },
    ],
}

# Topic keywords mapped to domain keys.
_KEYWORD_DOMAIN_MAP: dict[str, str] = {
    "quantum": "physics",
    "resonance": "physics",
    "acoustic": "physics",
    "frequency": "physics",
    "wave": "physics",
    "thermodynamic": "physics",
    "gravity": "physics",
    "electro": "physics",
    "gene": "biology",
    "protein": "biology",
    "cell": "biology",
    "neural": "biology",
    "evolution": "biology",
    "algorithm": "computer_science",
    "complexity": "computer_science",
    "distributed": "computer_science",
    "compiler": "computer_science",
    "runtime": "computer_science",
    "machine learning": "computer_science",
}


def _detect_domain(topic: str) -> str:
    """Infer the primary domain from the paper topic string."""
    topic_lower = topic.lower()
    for keyword, domain in _KEYWORD_DOMAIN_MAP.items():
        if keyword in topic_lower:
            return domain
    return "default"


def generate_reviewer_personas(
    topic: str,
    count: int = 4,
) -> list[dict[str, str]]:
    """Build adversarial reviewer personas tailored to *topic*.

    Returns up to *count* persona dicts, each with keys:
      name, role, focus, attack, system_prompt
    """
    domain = _detect_domain(topic)
    pool = list(_DOMAIN_PERSONAS.get(domain, _DOMAIN_PERSONAS["default"]))

    # Always mix in at least one cross-domain skeptic when the pool is domain-specific.
    if domain != "default":
        pool.append(_DOMAIN_PERSONAS["default"][0])  # Methodological Skeptic

    selected = pool[:count]
    # Guarantee the cross-domain skeptic is included if we added one.
    if domain != "default" and len(pool) > count and pool[-1] not in selected:
        selected[-1] = pool[-1]

    personas: list[dict[str, str]] = []
    for idx, spec in enumerate(selected):
        name = f"Reviewer_{chr(65 + idx)}"  # Reviewer_A, Reviewer_B, ...
        system_prompt = _build_reviewer_system_prompt(name, spec, topic)
        personas.append(
            {
                "name": name,
                "role": spec["role"],
                "focus": spec["focus"],
                "attack": spec["attack"],
                "system_prompt": system_prompt,
            }
        )
    return personas


def _build_reviewer_system_prompt(name: str, spec: dict[str, str], topic: str) -> str:
    return f"""# IDENTITY
You are {name}, a {spec['role']}.

# MANDATE
You have been summoned to a blind adversarial peer review of a research document
on the topic: "{topic}".

Your sole purpose is to find flaws. You are NOT here to praise the work.

# DOMAIN FOCUS
{spec['focus']}

# ATTACK VECTOR
{spec['attack']}

# ANTI-SYCOPHANCY RULES (MANDATORY)
- NEVER start with praise or compliments.
- NEVER use phrases like "interesting work", "great effort", "well-written".
- Lead EVERY response with the most critical flaw you have identified.
- If another reviewer's critique is weak or wrong, say so directly.
- Assign a severity to each flaw: [CRITICAL], [MAJOR], [MINOR].
- If you find NO flaws, state "NO FLAWS FOUND" (this should be extremely rare).

# RESPONSE FORMAT
Each response must follow this structure:
1. FLAWS IDENTIFIED (numbered, with severity tags)
2. RESPONSE TO OTHER REVIEWERS (agree/disagree with specific critiques)
3. REMAINING CONCERNS (unresolved issues from prior rounds)

Keep responses focused and under 200 words per turn."""


# ---------------------------------------------------------------------------
# 2. Swarm Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class SwarmConfig:
    """Tunable parameters for a peer-review swarm session."""

    rounds: int = 6
    max_tokens_per_turn: int = 512
    max_context_tokens: int = 6_000
    max_task_budget: float | None = None
    temperature: float = 0.4
    model_name: str = ""
    base_url: str = ""
    api_key: str = "not-needed"
    session_id: str = ""
    timeout: float = 120.0


@dataclass
class SwarmTranscript:
    """Immutable record of one completed swarm debate."""

    session_id: str
    topic: str
    document_hash: str
    personas: list[dict[str, str]]
    turns: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    total_duration_s: float = 0.0


@dataclass
class SwarmRuntimeState:
    """Mutable runtime state shared across all LLM calls in a task."""

    session_id: str
    cost_monitor: CostMonitor
    max_task_budget: float
    budget_prompt: BudgetPromptHandler | None = None


@dataclass
class AgentToolScope:
    """Manifest-defined tool access scopes for one specialist."""

    allowed: list[str] = field(default_factory=list)


@dataclass
class AgentMemoryRouting:
    """Manifest-defined memory and RAG routing hints."""

    categories: list[str] = field(default_factory=list)
    session_id: str | None = None


@dataclass
class AgentIdentity:
    """Strict agent identity contract sourced from manifest."""

    agent_id: str
    display_name: str
    role: str
    system_prompt: str


@dataclass
class AgentManifest:
    """Schema-first manifest replacing loose *_SOUL.md files."""

    schema_version: str
    identity: AgentIdentity
    tools: AgentToolScope = field(default_factory=AgentToolScope)
    memory: AgentMemoryRouting = field(default_factory=AgentMemoryRouting)


def load_agent_manifest(manifest_path: str | Path) -> AgentManifest:
    """Load a strict TOML agent manifest from disk."""
    path = Path(manifest_path)
    if tomllib is None:
        raise RuntimeError("tomllib unavailable; Python 3.11+ required for TOML manifests")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    identity = raw.get("identity", {})
    return AgentManifest(
        schema_version=str(raw.get("schema_version", "")),
        identity=AgentIdentity(
            agent_id=str(identity.get("id", "")),
            display_name=str(identity.get("display_name", "")),
            role=str(identity.get("role", "")),
            system_prompt=str(identity.get("system_prompt", "")),
        ),
        tools=AgentToolScope(allowed=list(raw.get("tools", {}).get("allowed", []))),
        memory=AgentMemoryRouting(
            categories=list(raw.get("memory", {}).get("categories", [])),
            session_id=raw.get("memory", {}).get("session_id"),
        ),
    )


def _workspace_root() -> Path:
    return Path(os.environ.get("JAMES_LIBRARY_PATH", Path.cwd())).resolve()


def _resolve_max_task_budget(config_budget: float | None) -> float:
    if config_budget is not None:
        return max(0.0, float(config_budget))

    for env_name in ("JAMES_MAX_TASK_BUDGET", "RAIN_MAX_TASK_BUDGET"):
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        try:
            return max(0.0, float(raw))
        except ValueError:
            logger.warning("Ignoring invalid %s value: %s", env_name, raw)

    return 1.00


def _build_runtime_state(
    *,
    session_id: str,
    config: SwarmConfig,
    workspace_root: Path,
    budget_prompt: BudgetPromptHandler | None = None,
) -> SwarmRuntimeState:
    return SwarmRuntimeState(
        session_id=session_id,
        cost_monitor=CostMonitor(session_id=session_id, workspace_root=workspace_root),
        max_task_budget=_resolve_max_task_budget(config.max_task_budget),
        budget_prompt=budget_prompt,
    )


def _default_budget_prompt(error: BudgetExceededError, current_limit: float) -> float | None:
    while True:
        answer = input(
            f"⚠️ BUDGET LIMIT REACHED (${current_limit:.2f}). "
            f"Total spent: ${error.total_spent:.2f}. "
            "Increase limit or terminate task? (y/n/limit)"
        ).strip().lower()

        if answer in {"", "n", "no"}:
            return None
        if answer in {"y", "yes"}:
            return round(max(current_limit + 1.0, error.total_spent + 1.0), 2)
        if answer == "limit":
            raw_limit = input("Enter new budget limit in USD: ").strip()
            try:
                new_limit = float(raw_limit)
            except ValueError:
                continue
            if new_limit > error.total_spent:
                return new_limit


def _handle_budget_exceeded(runtime_state: SwarmRuntimeState, error: BudgetExceededError) -> None:
    prompt = runtime_state.budget_prompt or _default_budget_prompt
    new_limit = prompt(error, runtime_state.max_task_budget)
    if new_limit is None:
        raise error
    runtime_state.max_task_budget = float(new_limit)
    logger.warning(
        "[COST] budget increased to $%.2f after overrun at $%.4f",
        runtime_state.max_task_budget,
        error.total_spent,
    )


def _response_token_usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")

    if usage is None:
        return 0, 0

    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)

    return max(0, int(prompt_tokens or 0)), max(0, int(completion_tokens or 0))


def _build_specialist_user_message(
    *,
    query: str,
    manifest: AgentManifest,
    room_context: str,
    prefetch_context: str = "",
) -> str:
    """Assemble a first-turn worker prompt with optional IDE prefetch context."""

    memory_hint = ", ".join(manifest.memory.categories) if manifest.memory.categories else "default"
    tool_hint = ", ".join(manifest.tools.allowed) if manifest.tools.allowed else "none"
    sections = [
        room_context,
        f"User query: {query}",
        f"Your role: {manifest.identity.role}",
        f"Allowed tools: {tool_hint}",
        f"Memory routes: {memory_hint}",
    ]
    if prefetch_context:
        sections.insert(1, prefetch_context)
    sections.append("Provide findings, assumptions, and one recommended next step.")
    return "\n\n".join(sections)


def _compact_messages_for_llm(
    messages: list[dict[str, str]],
    *,
    max_context_tokens: int,
    model: str = "cl100k_base",
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Compact transient prompt buffers before each LLM call."""

    result = context_manager.compact_history(messages, max_tokens=max_context_tokens, model=model)
    log_message = (
        f"[CONTEXT] before={result.original_tokens} "
        f"after={result.compacted_tokens} "
        f"saved={result.tokens_saved} "
        f"summaries={result.summary_count} "
        f"pruned={result.pruned_count}"
    )
    logger.debug(log_message)
    return result.compacted_messages, {
        "original_tokens": result.original_tokens,
        "compacted_tokens": result.compacted_tokens,
        "tokens_saved": result.tokens_saved,
        "summary_count": result.summary_count,
        "pruned_count": result.pruned_count,
        "log_message": log_message,
    }


def _chunk_text_as_messages(
    text: str,
    *,
    prefix: str,
    chunk_chars: int = 2_200,
) -> list[dict[str, str]]:
    """Split large prompt sections into independent user messages for compaction."""

    payload = text.strip()
    if not payload:
        return []
    if len(payload) <= chunk_chars:
        return [{"role": "user", "content": f"{prefix}\n{payload}"}]

    chunks = [
        payload[index : index + chunk_chars]
        for index in range(0, len(payload), chunk_chars)
    ]
    return [
        {"role": "user", "content": f"{prefix} (chunk {index}/{len(chunks)})\n{chunk}"}
        for index, chunk in enumerate(chunks, start=1)
    ]


async def run_blackboard_lab(
    query: str,
    manifests: list[AgentManifest],
    config: SwarmConfig | None = None,
    runtime_state: SwarmRuntimeState | None = None,
) -> dict[str, Any]:
    """Prototype blackboard orchestrator for multi-specialist collaboration.

    Each specialist receives the same shared room context plus a role-specific
    sub-task. The output is synthesized into a single response envelope.
    """
    cfg = config or SwarmConfig(rounds=1, temperature=0.3, max_tokens_per_turn=384)
    model = cfg.model_name or os.environ.get("LM_STUDIO_MODEL", "qwen2.5-coder-7b-instruct")
    base_url = cfg.base_url or os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    api_key = cfg.api_key or os.environ.get("LM_STUDIO_API_KEY", "not-needed")

    try:
        import openai as _openai
    except ImportError as exc:
        raise RuntimeError("openai package required for lab orchestration: pip install openai") from exc

    try:
        import httpx
        timeout = httpx.Timeout(min(15.0, cfg.timeout), read=cfg.timeout, write=15.0, connect=15.0)
        client = _openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    except ImportError:
        client = _openai.OpenAI(base_url=base_url, api_key=api_key)

    workspace_root = _workspace_root()
    session_id = runtime_state.session_id if runtime_state is not None else (
        cfg.session_id or f"blackboard_{uuid.uuid4().hex[:12]}"
    )
    runtime_state = runtime_state or _build_runtime_state(
        session_id=session_id,
        config=cfg,
        workspace_root=workspace_root,
    )
    prefetch_context = prefetch.build_prefetch_context(query, workspace_root)
    room_context = (
        "You are operating in a shared lab blackboard. "
        "Read peer notes and contribute only your specialist perspective."
    )
    per_agent_notes: list[dict[str, str]] = []

    for manifest in manifests:
        user_message = _build_specialist_user_message(
            query=query,
            manifest=manifest,
            room_context=room_context,
            prefetch_context=prefetch_context,
        )
        response = await _call_llm_async(
            client=client,
            model=model,
            messages=[
                {"role": "system", "content": manifest.identity.system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens_per_turn,
            max_context_tokens=cfg.max_context_tokens,
            runtime_state=runtime_state,
        )
        per_agent_notes.append(
            {
                "agent_id": manifest.identity.agent_id,
                "agent_name": manifest.identity.display_name,
                "role": manifest.identity.role,
                "notes": response,
            }
        )

    synthesis_messages = [
        {
            "role": "system",
            "content": "Synthesize into a concise multi-perspective answer with clear action items.",
        },
        {
            "role": "user",
            "content": (
                "You are the lab chair. Synthesize specialist notes into one integrated answer.\n\n"
                f"User query: {query}"
            ),
        },
        *(
            _chunk_text_as_messages(
                json.dumps(per_agent_notes, ensure_ascii=False, indent=2),
                prefix="[Specialist notes]",
            )
        ),
        {"role": "user", "content": "Synthesize the specialist notes into one integrated answer."},
    ]
    synthesis = await _call_llm_async(
        client=client,
        model=model,
        messages=synthesis_messages,
        temperature=0.2,
        max_tokens=768,
        max_context_tokens=cfg.max_context_tokens,
        runtime_state=runtime_state,
    )

    return {"query": query, "specialist_notes": per_agent_notes, "synthesized_response": synthesis}


async def _call_llm_async(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    max_context_tokens: int,
    runtime_state: SwarmRuntimeState | None = None,
) -> str:
    """Non-blocking LLM call via asyncio executor (openai client is sync)."""
    if runtime_state is not None:
        try:
            runtime_state.cost_monitor.check_budget(runtime_state.max_task_budget)
        except BudgetExceededError as error:
            _handle_budget_exceeded(runtime_state, error)

    compacted_messages, _meta = _compact_messages_for_llm(
        messages,
        max_context_tokens=max_context_tokens,
        model=model,
    )
    loop = asyncio.get_running_loop()

    def _sync_call() -> tuple[str, int, int]:
        response = client.chat.completions.create(
            model=model,
            messages=compacted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        prompt_tokens, completion_tokens = _response_token_usage(response)
        return response.choices[0].message.content.strip(), prompt_tokens, completion_tokens

    content, prompt_tokens, completion_tokens = await loop.run_in_executor(None, _sync_call)

    if runtime_state is not None:
        delta = runtime_state.cost_monitor.update_cost(
            model_name=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        logger.info(
            "[COST] session_total=$%.4f (+ $%.4f)",
            runtime_state.cost_monitor.session_cost,
            delta,
        )
        try:
            runtime_state.cost_monitor.check_budget(runtime_state.max_task_budget)
        except BudgetExceededError as error:
            _handle_budget_exceeded(runtime_state, error)

    return content


async def run_swarm(
    document: str,
    topic: str,
    config: SwarmConfig | None = None,
    runtime_state: SwarmRuntimeState | None = None,
) -> SwarmTranscript:
    """Run an isolated adversarial peer-review swarm.

    Args:
        document: The full markdown text of the paper to review.
        topic: Short description of the paper's subject area.
        config: Optional tuning knobs. Defaults are sane for local LM Studio.

    Returns:
        A SwarmTranscript containing the full debate record.
    """
    cfg = config or SwarmConfig()

    # Resolve model/endpoint from env if not provided.
    model = cfg.model_name or os.environ.get("LM_STUDIO_MODEL", "qwen2.5-coder-7b-instruct")
    base_url = cfg.base_url or os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    api_key = cfg.api_key or os.environ.get("LM_STUDIO_API_KEY", "not-needed")

    # Lazy import so the module can be loaded without openai installed.
    try:
        import openai as _openai
    except ImportError as exc:
        raise RuntimeError("openai package required for swarm orchestration: pip install openai") from exc

    try:
        import httpx
        timeout = httpx.Timeout(min(15.0, cfg.timeout), read=cfg.timeout, write=15.0, connect=15.0)
        client = _openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    except ImportError:
        client = _openai.OpenAI(base_url=base_url, api_key=api_key)

    # Generate adversarial personas.
    personas = generate_reviewer_personas(topic, count=4)

    session_id = runtime_state.session_id if runtime_state is not None else (
        cfg.session_id or f"swarm_{uuid.uuid4().hex[:12]}"
    )
    runtime_state = runtime_state or _build_runtime_state(
        session_id=session_id,
        config=cfg,
        workspace_root=_workspace_root(),
    )
    doc_hash = f"{len(document)}_{hash(document) & 0xFFFFFFFF:08x}"
    t_start = time.monotonic()

    transcript = SwarmTranscript(
        session_id=session_id,
        topic=topic,
        document_hash=doc_hash,
        personas=[{k: v for k, v in p.items() if k != "system_prompt"} for p in personas],
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    # Truncate document to avoid context overflow on small models.
    max_doc_chars = 12_000
    doc_excerpt = document[:max_doc_chars]
    if len(document) > max_doc_chars:
        doc_excerpt += "\n\n[... DOCUMENT TRUNCATED FOR REVIEW ...]"

    # Build the shared document context (injected once per turn).
    doc_context = (
        f"# DOCUMENT UNDER REVIEW\n"
        f"Topic: {topic}\n\n"
        f"{doc_excerpt}"
    )

    debate_messages: list[dict[str, str]] = [{"role": "user", "content": doc_context}]

    for round_num in range(1, cfg.rounds + 1):
        for persona in personas:
            user_msg = (
                f"# YOUR TURN (Round {round_num}/{cfg.rounds})\n"
                f"Provide your critique for this round. "
                f"Address other reviewers' points where relevant."
            )

            try:
                response_text = await _call_llm_async(
                    client=client,
                    model=model,
                    messages=[
                        {"role": "system", "content": persona["system_prompt"]},
                        *debate_messages,
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens_per_turn,
                    max_context_tokens=cfg.max_context_tokens,
                    runtime_state=runtime_state,
                )
            except BudgetExceededError:
                raise
            except Exception as e:
                response_text = f"[ERROR: {type(e).__name__}: {e}]"
                logger.warning("Swarm LLM call failed for %s: %s", persona["name"], e)

            turn_record = {
                "round": round_num,
                "reviewer": persona["name"],
                "role": persona["role"],
                "content": response_text,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            transcript.turns.append(turn_record)
            debate_messages.append(
                {
                    "role": "user",
                    "content": f"[{persona['name']} ({persona['role']})] {response_text}",
                }
            )

    t_end = time.monotonic()
    transcript.finished_at = datetime.now(timezone.utc).isoformat()
    transcript.total_duration_s = round(t_end - t_start, 2)

    return transcript


# ---------------------------------------------------------------------------
# 3. Synthesizer - compress debate into structured report
# ---------------------------------------------------------------------------

_SYNTHESIZER_SYSTEM_PROMPT = """You are the Peer Review Synthesizer.

You receive the raw transcript of an adversarial multi-reviewer debate about a
research document. Your job is to compress it into a clear, actionable report.

# OUTPUT FORMAT (Markdown, use these exact headings)

## Executive Summary
One paragraph: overall assessment and confidence level.

## Critical Flaws
Numbered list. Each item: description, which reviewer(s) raised it, severity.

## Mathematical / Logical Errors
Numbered list of specific errors with page/section references where possible.

## Methodological Concerns
Issues with experimental design, statistical analysis, or reproducibility.

## Points of Reviewer Consensus
Critiques that multiple reviewers independently agreed on (strongest signal).

## Points of Reviewer Disagreement
Where reviewers contradicted each other (requires author judgment).

## Suggested Revisions
Prioritized action items for the author, ordered by severity.

## Reviewer Confidence Scores
For each reviewer, estimate how confident/substantiated their critiques were (1-5).

# RULES
- Be ruthlessly concise. No filler.
- Preserve severity tags: [CRITICAL], [MAJOR], [MINOR].
- If reviewers raised the same flaw independently, flag it as HIGH CONFIDENCE.
- Do NOT add new critiques. Only synthesize what reviewers actually said."""


async def synthesize_report(
    transcript: SwarmTranscript,
    config: SwarmConfig | None = None,
    runtime_state: SwarmRuntimeState | None = None,
) -> str:
    """Compress a swarm transcript into a structured Peer_Review_Report.

    Returns the report as a markdown string.
    """
    cfg = config or SwarmConfig()
    runtime_state = runtime_state or _build_runtime_state(
        session_id=transcript.session_id,
        config=cfg,
        workspace_root=_workspace_root(),
    )
    model = cfg.model_name or os.environ.get("LM_STUDIO_MODEL", "qwen2.5-coder-7b-instruct")
    base_url = cfg.base_url or os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    api_key = cfg.api_key or os.environ.get("LM_STUDIO_API_KEY", "not-needed")

    try:
        import openai as _openai
    except ImportError as exc:
        raise RuntimeError("openai package required: pip install openai") from exc

    try:
        import httpx
        timeout = httpx.Timeout(15.0, read=cfg.timeout, write=15.0, connect=15.0)
        client = _openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    except ImportError:
        client = _openai.OpenAI(base_url=base_url, api_key=api_key)

    # Build a condensed version of the debate for the synthesizer.
    debate_lines: list[str] = []
    for turn in transcript.turns:
        debate_lines.append(
            f"### [{turn['reviewer']} - {turn['role']}] (Round {turn['round']})\n"
            f"{turn['content']}\n"
        )
    debate_text = "\n".join(debate_lines)

    # Truncate if the debate is very long.
    max_debate_chars = 20_000
    if len(debate_text) > max_debate_chars:
        debate_text = debate_text[:max_debate_chars] + "\n\n[... DEBATE TRUNCATED ...]"

    report = await _call_llm_async(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": _SYNTHESIZER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"# PEER REVIEW SWARM TRANSCRIPT\n"
                    f"Topic: {transcript.topic}\n"
                    f"Session: {transcript.session_id}\n"
                    f"Reviewers: {len(transcript.personas)}\n"
                    f"Rounds: {transcript.turns[-1]['round'] if transcript.turns else 0}\n"
                    f"Duration: {transcript.total_duration_s}s"
                ),
            },
            *(_chunk_text_as_messages(debate_text, prefix="[Debate transcript]")),
            {
                "role": "user",
                "content": "# TASK\nSynthesize the above debate into the required report format.",
            },
        ],
        temperature=0.2,
        max_tokens=2048,
        max_context_tokens=cfg.max_context_tokens,
        runtime_state=runtime_state,
    )

    # Prepend metadata header.
    header = (
        f"# Peer Review Report\n\n"
        f"| Field | Value |\n"
        f"|-------|-------|\n"
        f"| Session | `{transcript.session_id}` |\n"
        f"| Topic | {transcript.topic} |\n"
        f"| Reviewers | {', '.join(p['name'] + ' (' + p['role'] + ')' for p in transcript.personas)} |\n"
        f"| Rounds | {transcript.turns[-1]['round'] if transcript.turns else 0} |\n"
        f"| Duration | {transcript.total_duration_s}s |\n"
        f"| Generated | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} |\n\n"
        f"---\n\n"
    )

    return header + report


# ---------------------------------------------------------------------------
# 4. invoke_peer_review - tool-callable entry point
# ---------------------------------------------------------------------------

async def invoke_peer_review(
    document: str,
    topic: str,
    rounds: int = 6,
    output_path: str | None = None,
    model_name: str = "",
    base_url: str = "",
) -> dict[str, Any]:
    """Callable tool for main R.A.I.N. Lab agents (James, Luca, etc.).

    Runs a full peer-review swarm in isolation and returns the synthesized
    report. The main agent's context is NOT polluted by the raw debate.

    Args:
        document: Full markdown text of the paper to review.
        topic: Short topic description for persona generation.
        rounds: Number of debate rounds (default 6).
        output_path: Optional path to write Peer_Review_Report.md.
        model_name: LLM model override (default: env / qwen2.5-coder-7b-instruct).
        base_url: LLM endpoint override (default: env / localhost:1234).

    Returns:
        Dict with keys: report (str), transcript_summary (dict), output_file (str|None).
    """
    cfg = SwarmConfig(
        rounds=max(3, min(rounds, 12)),  # Clamp to sane range.
        model_name=model_name,
        base_url=base_url,
    )
    session_id = cfg.session_id or f"swarm_{uuid.uuid4().hex[:12]}"
    runtime_state = _build_runtime_state(
        session_id=session_id,
        config=cfg,
        workspace_root=_workspace_root(),
    )

    # Phase 1: Run the adversarial debate.
    transcript = await run_swarm(
        document=document,
        topic=topic,
        config=cfg,
        runtime_state=runtime_state,
    )

    # Phase 2: Synthesize the report.
    report = await synthesize_report(
        transcript,
        config=cfg,
        runtime_state=runtime_state,
    )

    # Phase 3: Optionally persist to disk.
    output_file = None
    if output_path:
        out = Path(output_path)
    else:
        archive_dir = Path(os.environ.get("JAMES_LIBRARY_PATH", ".")) / "meeting_archives"
        archive_dir.mkdir(parents=True, exist_ok=True)
        out = archive_dir / f"Peer_Review_Report_{transcript.session_id}.md"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    output_file = str(out)
    logger.info("Peer review report written to %s", output_file)

    # Also persist the raw transcript as JSONL for auditability.
    transcript_path = out.parent / (out.stem + ".transcript.jsonl")
    with open(transcript_path, "w", encoding="utf-8") as f:
        for turn in transcript.turns:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")

    return {
        "report": report,
        "transcript_summary": {
            "session_id": transcript.session_id,
            "topic": transcript.topic,
            "reviewers": len(transcript.personas),
            "rounds": cfg.rounds,
            "total_turns": len(transcript.turns),
            "duration_s": transcript.total_duration_s,
        },
        "output_file": output_file,
    }


def invoke_peer_review_sync(
    document: str,
    topic: str,
    rounds: int = 6,
    output_path: str | None = None,
    model_name: str = "",
    base_url: str = "",
) -> dict[str, Any]:
    """Synchronous wrapper for invoke_peer_review.

    Use this from non-async contexts (e.g., the RLM tool injection layer).
    Creates a new event loop if none is running.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    coro = invoke_peer_review(
        document=document,
        topic=topic,
        rounds=rounds,
        output_path=output_path,
        model_name=model_name,
        base_url=base_url,
    )

    if loop and loop.is_running():
        # We're inside an existing event loop (e.g., Jupyter, async framework).
        # Schedule as a task and block via threading to avoid deadlock.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)
