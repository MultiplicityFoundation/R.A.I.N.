"""
graph_bridge.py  —  Safe wrapper for the Graph-R1 hypergraph retrieval engine.

If the `graph_r1_engine` submodule is missing or its dependencies aren't
installed, the rest of the application continues to run normally with
GRAPH_AVAILABLE = False.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Safe import – adds the local submodule to sys.path and attempts to pull in
# the two classes we need.  Any failure is caught silently.
# ---------------------------------------------------------------------------
GRAPH_AVAILABLE = False
GraphR1 = None
QueryParam = None

try:
    _engine_dir = str(Path(__file__).resolve().parent / "graph_r1_engine")
    if _engine_dir not in sys.path:
        sys.path.insert(0, _engine_dir)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from graphr1 import GraphR1 as _GraphR1, QueryParam as _QueryParam  # type: ignore

    GraphR1 = _GraphR1
    QueryParam = _QueryParam
    GRAPH_AVAILABLE = True

except Exception:
    # Missing folder, missing dependencies (faiss, FlagEmbedding, …), etc.
    pass


# ---------------------------------------------------------------------------
# Public wrapper class
# ---------------------------------------------------------------------------
class HypergraphManager:
    """
    High-level, crash-safe interface to Graph-R1.

    Usage::

        hg = HypergraphManager()
        hg.build_graph("/path/to/library")
        connections = hg.query("resonance patterns")
    """

    def __init__(self):
        self.available: bool = GRAPH_AVAILABLE
        self._engine = None          # GraphR1 instance (created in build_graph)
        self._indexed: bool = False

    # ---- indexing ----------------------------------------------------------
    def build_graph(self, library_path: str, verbose: bool = False) -> bool:
        """
        Scan *library_path* for .md / .txt files, feed them into Graph-R1's
        knowledge-graph builder, and return True on success.

        If Graph-R1 is unavailable the method returns False immediately
        without raising.
        """
        if not self.available:
            if verbose:
                print("⚠️  Graph-R1 not available — skipping hypergraph build.")
            return False

        lab = Path(library_path)
        if not lab.exists():
            if verbose:
                print(f"⚠️  Library path does not exist: {lab}")
            return False

        try:
            working_dir = str(lab / ".graphr1_cache")
            os.makedirs(working_dir, exist_ok=True)

            self._engine = GraphR1(working_dir=working_dir)

            # Collect text files (skip souls, logs, meeting files)
            skip_names = {"SOUL", "LOG", "MEETING"}
            docs: list[str] = []
            for fp in sorted(lab.rglob("*")):
                if not fp.is_file():
                    continue
                if fp.suffix.lower() not in {".md", ".txt"}:
                    continue
                if any(s in fp.name.upper() for s in skip_names):
                    continue
                try:
                    text = fp.read_text(encoding="utf-8", errors="ignore")
                    if text.strip():
                        docs.append(text)
                except Exception:
                    continue

            if not docs:
                if verbose:
                    print("⚠️  No documents found for graph indexing.")
                return False

            if verbose:
                print(f"🔗 Indexing {len(docs)} documents into hypergraph…")

            # Insert in batches to stay within API limits
            batch_size = 50
            for i in range(0, len(docs), batch_size):
                self._engine.insert(docs[i : i + batch_size])

            self._indexed = True
            if verbose:
                print("✅ Hypergraph index built successfully.")
            return True

        except Exception as exc:
            if verbose:
                print(f"⚠️  Hypergraph build failed: {exc}")
            self._indexed = False
            return False

    # ---- querying ----------------------------------------------------------
    def query(self, topic: str, depth: int = 2, verbose: bool = False) -> str:
        """
        Query the hypergraph for *topic* and return a human-readable block of
        hidden connections.  Returns an empty string when Graph-R1 is
        unavailable or the graph has not been built yet.

        Parameters
        ----------
        topic : str
            Natural-language query (e.g. a meeting topic).
        depth : int, optional
            Retrieval depth — maps to ``top_k`` in QueryParam.  Default is 2.
        verbose : bool, optional
            Print status messages.
        """
        if not self.available or not self._indexed or self._engine is None:
            return ""

        try:
            param = QueryParam(
                only_need_context=True,
                top_k=depth * 30,          # scale top-k with depth
                mode="hybrid",
            )
            result = self._engine.query(topic, param=param)

            if not result:
                return ""

            # Wrap the raw context in a labelled block for the LLM prompt
            header = f"### 🕸️ HYPERGRAPH CONNECTIONS (depth={depth})\n"
            return header + str(result)

        except Exception as exc:
            if verbose:
                print(f"⚠️  Hypergraph query failed: {exc}")
            return ""
