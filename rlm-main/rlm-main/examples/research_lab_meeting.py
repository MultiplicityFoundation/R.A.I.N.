"""Run a recursive, multi-agent style research lab over local markdown papers + web search.

This example configures RLM with:
- Persistent memory (`history_*`) across calls
- File-backed research memory (`memory/YYYY-MM-DD.md` + `MEMORY.md`)
- REPL tools for local paper retrieval (list/read/search)
- Provider-style web tools (search + fetch) with simple in-run caching
- A meeting-style orchestration prompt with explicit role fanout/fanin

Usage:
    uv run python examples/research_lab_meeting.py \
      --papers-dir ./papers \
      --query "Find a novel idea combining these papers"
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

SETUP_CODE = r"""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import re
import requests
from urllib.parse import quote_plus

PAPERS_DIR = Path(os.environ.get("RLM_PAPERS_DIR", "./papers")).expanduser().resolve()
MEMORY_DIR = Path(os.environ.get("RLM_MEMORY_DIR", "./memory")).expanduser().resolve()
MEMORY_FILE = Path(os.environ.get("RLM_LONG_MEMORY_FILE", "./MEMORY.md")).expanduser().resolve()

# In-run cache for deterministic repeated calls in a single execution.
_WEB_CACHE = {}


def _today_memory_file() -> Path:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return MEMORY_DIR / f"{day}.md"


def append_memory(note: str, section: str = "lab") -> str:
    stamp = datetime.now(timezone.utc).isoformat()
    target = _today_memory_file()
    entry = f"\n## [{stamp}] {section}\n\n{note.strip()}\n"
    with target.open("a", encoding="utf-8") as f:
        f.write(entry)
    return str(target)


def read_recent_memory(max_chars: int = 4000) -> str:
    target = _today_memory_file()
    if not target.exists():
        return ""
    text = target.read_text(encoding="utf-8")
    return text[-max_chars:]


def update_long_memory(note: str) -> str:
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    old = MEMORY_FILE.read_text(encoding="utf-8") if MEMORY_FILE.exists() else "# Long-Term Memory\n"
    stamp = datetime.now(timezone.utc).isoformat()
    with MEMORY_FILE.open("w", encoding="utf-8") as f:
        f.write(old.rstrip() + f"\n\n## {stamp}\n\n{note.strip()}\n")
    return str(MEMORY_FILE)


def list_papers() -> list[str]:
    if not PAPERS_DIR.exists():
        return []
    return sorted([str(p.relative_to(PAPERS_DIR)) for p in PAPERS_DIR.rglob("*.md")])


def read_paper(path: str) -> str:
    p = (PAPERS_DIR / path).resolve()
    if PAPERS_DIR not in p.parents and p != PAPERS_DIR:
        raise ValueError("Path escapes paper directory")
    return p.read_text(encoding="utf-8")


def search_library(query: str, max_results: int = 5) -> list[dict]:
    results = []
    q = query.lower()
    for rel_path in list_papers():
        text = read_paper(rel_path)
        score = text.lower().count(q)
        if score > 0:
            preview = text[:600].replace("\n", " ")
            results.append({"paper": rel_path, "score": score, "preview": preview})
    return sorted(results, key=lambda x: x["score"], reverse=True)[:max_results]


def semantic_search(query: str, max_results: int = 5) -> list[dict]:
    papers = list_papers()
    if not papers:
        return []

    prompts = []
    paper_names = []
    for rel_path in papers:
        text = read_paper(rel_path)
        sample = text[:12000]
        prompts.append(
            "Score 0-10 relevance for this query based on excerpt. "
            "Return JSON only with keys score and reason.\\n"
            f"QUERY: {query}\\n"
            f"PAPER: {rel_path}\\n"
            f"EXCERPT:\\n{sample}"
        )
        paper_names.append(rel_path)

    raws = llm_query_batched(prompts)
    ranked = []
    for rel_path, raw in zip(paper_names, raws):
        try:
            parsed = json.loads(raw)
            score = float(parsed.get("score", 0))
            reason = str(parsed.get("reason", ""))
        except Exception:
            score = 0.0
            reason = f"Could not parse model response: {str(raw)[:200]}"
        ranked.append({"paper": rel_path, "score": score, "reason": reason})

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:max_results]


def web_search(query: str, max_results: int = 5, provider: str = "duckduckgo") -> list[dict]:
    # Web search with provider-like interface.
    # Current provider support:
    # - duckduckgo: HTML search page parsing (no API key)
    cache_key = f"search::{provider}::{query}::{max_results}"
    if cache_key in _WEB_CACHE:
        return _WEB_CACHE[cache_key]

    if provider != "duckduckgo":
        raise ValueError("Unsupported provider. Use provider='duckduckgo'.")

    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    html = resp.text

    # Parse anchor tags used by DuckDuckGo HTML results.
    results = []
    for match in re.finditer(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html):
        href = match.group(1)
        title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
        if title and href:
            results.append({"title": title, "url": href, "provider": provider})
        if len(results) >= max_results:
            break

    _WEB_CACHE[cache_key] = results
    return results


def web_fetch(url: str, max_chars: int = 8000) -> str:
    cache_key = f"fetch::{url}::{max_chars}"
    if cache_key in _WEB_CACHE:
        return _WEB_CACHE[cache_key]

    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    text = resp.text
    # Lightweight readability fallback.
    clean = re.sub(r"<script[\\s\\S]*?</script>", "", text, flags=re.IGNORECASE)
    clean = re.sub(r"<style[\\s\\S]*?</style>", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"<[^>]+>", " ", clean)
    clean = re.sub(r"\\s+", " ", clean).strip()
    out = clean[:max_chars]
    _WEB_CACHE[cache_key] = out
    return out


def search_web(query: str, max_results: int = 5) -> list[dict]:
    # Backward-compatible alias.
    return web_search(query=query, max_results=max_results, provider="duckduckgo")
"""


def build_meeting_prompt(user_query: str) -> str:
    return f"""
You are running a recursive research lab meeting to answer this request:
{user_query}

Strict workflow:
1) Bootstrap evidence:
   - Call list_papers(), search_library(), and semantic_search().
   - Read top papers with read_paper().
   - Search web with web_search() / search_web(); fetch key pages with web_fetch().
2) Build role briefs and run role analyses in parallel with llm_query_batched:
   - Agent A: Literature Synthesizer
   - Agent B: Skeptical Reviewer
   - Agent C: Novelty Architect
3) Reconcile disagreements in a chairperson step using llm_query.
4) Write durable notes using append_memory().
5) If a high-signal enduring insight appears, also call update_long_memory().
6) Return FINAL(...) only after synthesis is complete.

Output must include:
- Consensus summary
- 3+ novel recursive ideas combining multiple papers
- Explicit evidence table (paper/file/url -> claim)
- Risks/failure modes + next experiments
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursive multi-agent research lab over markdown papers"
    )
    parser.add_argument("--query", required=True, help="Research question for the lab meeting")
    parser.add_argument(
        "--papers-dir",
        default="./papers",
        help="Directory containing markdown papers (*.md)",
    )
    parser.add_argument(
        "--memory-dir",
        default="./memory",
        help="Directory where daily memory markdown files are stored",
    )
    parser.add_argument(
        "--long-memory-file",
        default="./MEMORY.md",
        help="Path to long-term curated memory markdown file",
    )
    parser.add_argument("--model", default="gpt-5-nano", help="Model name for backend")
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--max-depth", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")

    os.environ["RLM_PAPERS_DIR"] = args.papers_dir
    os.environ["RLM_MEMORY_DIR"] = args.memory_dir
    os.environ["RLM_LONG_MEMORY_FILE"] = args.long_memory_file

    logger = RLMLogger(log_dir="./logs")

    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": args.model,
            "api_key": api_key,
        },
        environment="local",
        environment_kwargs={
            "setup_code": SETUP_CODE,
            "persistent": True,
        },
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        logger=logger,
        verbose=True,
    )

    completion = rlm.completion(build_meeting_prompt(args.query), root_prompt=args.query)
    print("\n=== FINAL OUTPUT ===\n")
    print(completion.response)


if __name__ == "__main__":
    main()
