"""
graph_engine.py
---------------
Graph Math Engine — Part 3 of the Git ingestion pipeline.

Responsibilities:
  - Maintain a directed dependency graph (networkx.DiGraph) that evolves
    commit-by-commit as parsed file data arrives from CodeParser.
  - Calculate architectural health metrics (score, complexity, cycles).
  - Export the graph state in a React Flow-compatible JSON format,
    annotating cyclic edges with is_violation=True.

Data flow:
  GitWalker  →  CodeParser  →  RepoGraph  →  FastAPI / React Front-end
"""

from __future__ import annotations

import logging
import math
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants & tunables
# ---------------------------------------------------------------------------

# simple_cycles() is bounded to cycles of at most this many nodes.
# Prevents O(n!) explosion on dense graphs while still catching all
# real-world circular imports (which are almost always short).
_MAX_CYCLE_LENGTH: int = 10

# Penalty deducted from the overall score per detected cycle.
_PENALTY_PER_CYCLE: float = 5.0

# Threshold above which a single node's complexity is "extreme".
_EXTREME_COMPLEXITY_THRESHOLD: int = 20

# Penalty per node that exceeds the extreme complexity threshold.
_PENALTY_PER_EXTREME_NODE: float = 2.0

# React Flow uses pixel coordinates; we space nodes on a simple grid.
_GRID_SPACING_X: int = 220
_GRID_SPACING_Y: int = 130


# ---------------------------------------------------------------------------
# RepoGraph
# ---------------------------------------------------------------------------

class RepoGraph:
    """
    Directed dependency graph of a repository that evolves over time.

    Each **node** represents a source file path.
    Each **edge** ``(A → B)`` means "file A imports file B."

    Node attributes
    ---------------
    complexity_score : int
        Sum of cyclomatic complexities of all functions/classes in the file.
    functions : list[dict]
        Raw function records from CodeParser (name, kind, start_line, …).
    last_commit : str
        Hash of the most recent commit that touched this file.

    Usage
    -----
    >>> rg = RepoGraph()
    >>> rg.update_commit_state("abc123", {"src/app.py": parser_result, ...})
    >>> metrics = rg.calculate_metrics()
    >>> payload = rg.export_graph_state()
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()

        # Cache invalidated whenever the graph mutates; recomputed lazily.
        self._cycle_cache: list[list[str]] | None = None
        # Set of (source, target) edge tuples involved in any cycle.
        self._violation_edges: set[tuple[str, str]] = set()
        # Bus factor per module, updated commit by commit
        self._module_bus_factors: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public — graph mutation
    # ------------------------------------------------------------------

        self,
        commit_hash: str,
        parsed_files: dict[str, dict[str, Any]],
        module_bus_factors: dict[str, float] | None = None,
    ) -> None:
        """
        Ingest the parsed output of one commit into the graph.

        For every file in *parsed_files*:

        1. **Ensure the node exists** (add it if new).
        2. **Remove all existing outgoing edges** from that node — the
           import list may have changed entirely since the last commit.
        3. **Update node attributes** — complexity_score and raw function list.
        4. **Add new directed edges** ``file → imported_module`` for every
           import discovered by CodeParser.

        Parameters
        ----------
        commit_hash : str
            The commit SHA-1 that produced this set of parsed files.
        module_bus_factors : dict[str, float] | None
            ``{module_name: bus_factor_score, ...}``
        """
        if module_bus_factors is not None:
            self._module_bus_factors = module_bus_factors

        if not parsed_files:
            return

        for file_path, parse_result in parsed_files.items():
            functions: list[dict] = parse_result.get("functions", [])
            imports:   list[str]  = parse_result.get("imports",   [])

            # ── Step 1: ensure node ─────────────────────────────────────
            if not self._graph.has_node(file_path):
                self._graph.add_node(file_path)

            # ── Step 2: drop stale outgoing edges ───────────────────────
            # list() is required — can't mutate while iterating the view
            stale_successors = list(self._graph.successors(file_path))
            for successor in stale_successors:
                self._graph.remove_edge(file_path, successor)

            # ── Step 3: update node attributes ──────────────────────────
            complexity_score: int = sum(
                fn.get("complexity", 1) for fn in functions
            )
            self._graph.nodes[file_path].update({
                "complexity_score": complexity_score,
                "functions":        functions,
                "last_commit":      commit_hash,
            })

            # ── Step 4: add new dependency edges ────────────────────────
            for imported_module in imports:
                # Ensure the target node exists even if we haven't parsed
                # it yet (it may be an external package or a not-yet-seen file).
                if not self._graph.has_node(imported_module):
                    self._graph.add_node(imported_module, complexity_score=0,
                                         functions=[], last_commit=None)
                self._graph.add_edge(file_path, imported_module)

        # Invalidate the cycle cache — graph topology has changed.
        self._cycle_cache    = None
        self._violation_edges = set()

    # ------------------------------------------------------------------
    # Public — metrics
    # ------------------------------------------------------------------

    def calculate_metrics(self) -> dict[str, Any]:
        """
        Compute architectural health metrics for the current graph state.

        Returns
        -------
        dict with keys:

        overall_score : float
            Health score in [0, 100].  Starts at 100 and is penalised by:
            - ``PENALTY_PER_CYCLE`` for each detected dependency cycle
            - ``PENALTY_PER_EXTREME_NODE`` for each node whose complexity
              exceeds ``EXTREME_COMPLEXITY_THRESHOLD``
        complexity_total : int
            Sum of ``complexity_score`` across all nodes.
        dependency_cycles : int
            Number of distinct circular dependency chains detected.
            Cycle search is bounded to ``_MAX_CYCLE_LENGTH`` nodes to
            prevent combinatorial explosion on large graphs.
        node_count : int
        edge_count : int
        """
        cycles      = self._get_cycles()          # cached
        cycle_count = len(cycles)

        complexity_scores: list[int] = [
            data.get("complexity_score", 0)
            for _, data in self._graph.nodes(data=True)
        ]
        complexity_total = sum(complexity_scores)

        extreme_nodes = sum(
            1 for c in complexity_scores if c > _EXTREME_COMPLEXITY_THRESHOLD
        )

        raw_score = (
            100.0
            - (cycle_count   * _PENALTY_PER_CYCLE)
            - (extreme_nodes * _PENALTY_PER_EXTREME_NODE)
        )
        overall_score = max(0.0, min(100.0, raw_score))

        return {
            "overall_score":      round(overall_score, 2),
            "complexity_total":   complexity_total,
            "dependency_cycles":  cycle_count,
            "node_count":         self._graph.number_of_nodes(),
            "edge_count":         self._graph.number_of_edges(),
        }

    # ------------------------------------------------------------------
    # Public — React Flow export
    # ------------------------------------------------------------------

    def export_graph_state(self) -> dict[str, Any]:
        """
        Serialise the current graph into a React Flow-compatible payload.

        Node format
        -----------
        .. code-block:: json

            {
                "id":       "src/app.py",
                "type":     "default",
                "data": {
                    "label":            "app.py",
                    "complexity_score": 12,
                    "last_commit":      "abc123",
                    "function_count":   4
                },
                "position": {"x": 440, "y": 130}
            }

        Edge format
        -----------
        .. code-block:: json

            {
                "id":           "src/app.py->os",
                "source":       "src/app.py",
                "target":       "os",
                "is_violation": false,
                "animated":     false
            }

        Returns
        -------
        dict with ``"nodes"`` and ``"edges"`` lists.
        """
        # Ensure violation set is populated
        self._get_cycles()

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        # ── Nodes ───────────────────────────────────────────────────────
        # Lay out nodes on a simple grid.  React Flow handles pretty
        # auto-layout on the front-end; we just need non-overlapping seeds.
        all_nodes = list(self._graph.nodes(data=True))
        cols = max(1, math.ceil(math.sqrt(len(all_nodes))))

        for idx, (node_id, attrs) in enumerate(all_nodes):
            row = idx // cols
            col = idx %  cols
            label = node_id.split("/")[-1] if "/" in node_id else node_id

            nodes.append({
                "id":   node_id,
                "type": "default",
                "data": {
                    "label":            label,
                    "full_path":        node_id,
                    "group":            node_id.split("/")[0] if "/" in node_id else "root",
                    "complexity_score": attrs.get("complexity_score", 0),
                    "last_commit":      attrs.get("last_commit"),
                    "function_count":   len(attrs.get("functions", [])),
                    "bus_factor":       self._module_bus_factors.get(node_id.split("/")[0] if "/" in node_id else "root", 10.0),
                },
                "position": {
                    "x": col * _GRID_SPACING_X,
                    "y": row * _GRID_SPACING_Y,
                },
            })

        # ── Edges ────────────────────────────────────────────────────────
        for src, tgt in self._graph.edges():
            is_violation = (src, tgt) in self._violation_edges
            edges.append({
                "id":           f"{src}->{tgt}",
                "source":       src,
                "target":       tgt,
                "is_violation": is_violation,
                # React Flow renders animated dashed lines for violations
                "animated":     is_violation,
                "style": {
                    "stroke": "#ef4444" if is_violation else "#6366f1",
                },
            })

        return {"nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def graph(self) -> nx.DiGraph:
        """Direct access to the underlying DiGraph (read-only use advised)."""
        return self._graph

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    # ------------------------------------------------------------------
    # Private — cycle detection
    # ------------------------------------------------------------------

    def _get_cycles(self) -> list[list[str]]:
        """
        Return the cached list of dependency cycles.

        Uses ``nx.simple_cycles(G, length_bound=_MAX_CYCLE_LENGTH)`` to
        enumerate all simple cycles up to ``_MAX_CYCLE_LENGTH`` nodes long.
        This is O(n + e) per cycle found — safe on graphs with hundreds of
        nodes, and bounded against pathological cases.

        The result is cached until the next call to
        :meth:`update_commit_state`.
        """
        if self._cycle_cache is not None:
            return self._cycle_cache

        try:
            raw_cycles = list(
                nx.simple_cycles(self._graph, length_bound=_MAX_CYCLE_LENGTH)
            )
        except Exception as exc:          # noqa: BLE001
            logger.warning("Cycle detection failed: %s", exc)
            raw_cycles = []

        self._cycle_cache = raw_cycles

        # Rebuild violation-edge set from fresh cycles
        self._violation_edges = set()
        for cycle in raw_cycles:
            # A cycle [a, b, c] has edges a→b, b→c, c→a
            for i, node in enumerate(cycle):
                nxt = cycle[(i + 1) % len(cycle)]
                self._violation_edges.add((node, nxt))

        logger.debug(
            "Cycle detection complete: %d cycle(s), %d violation edge(s).",
            len(raw_cycles),
            len(self._violation_edges),
        )
        return self._cycle_cache


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import io
    import json
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s  %(message)s")
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    # ── Simulate a two-commit history ────────────────────────────────────
    #
    # Commit 1: Initial structure
    #   app.py  imports  db.py, utils.py
    #   db.py   imports  utils.py
    #
    # Commit 2: Introduce a circular dependency
    #   utils.py now imports app.py  ← cycle: app → utils → app
    #   app.py   complexity grows

    commit1_data: dict[str, dict[str, Any]] = {
        "src/app.py": {
            "functions": [
                {"name": "main",       "kind": "function", "start_line": 1,  "end_line": 30, "complexity": 8},
                {"name": "setup",      "kind": "function", "start_line": 32, "end_line": 45, "complexity": 3},
            ],
            "imports": ["src/db.py", "src/utils.py"],
        },
        "src/db.py": {
            "functions": [
                {"name": "connect",    "kind": "function", "start_line": 1,  "end_line": 20, "complexity": 5},
                {"name": "query",      "kind": "function", "start_line": 22, "end_line": 55, "complexity": 12},
            ],
            "imports": ["src/utils.py"],
        },
        "src/utils.py": {
            "functions": [
                {"name": "parse_env",  "kind": "function", "start_line": 1,  "end_line": 10, "complexity": 2},
            ],
            "imports": [],
        },
    }

    commit2_data: dict[str, dict[str, Any]] = {
        "src/app.py": {
            "functions": [
                {"name": "main",       "kind": "function", "start_line": 1,  "end_line": 40, "complexity": 15},
                {"name": "setup",      "kind": "function", "start_line": 42, "end_line": 60, "complexity": 6},
                {"name": "App",        "kind": "class",    "start_line": 62, "end_line": 90, "complexity": 9},
            ],
            "imports": ["src/db.py", "src/utils.py"],
        },
        # utils.py now imports app.py → cycle introduced
        "src/utils.py": {
            "functions": [
                {"name": "parse_env",  "kind": "function", "start_line": 1,  "end_line": 10, "complexity": 2},
                {"name": "get_config", "kind": "function", "start_line": 12, "end_line": 25, "complexity": 4},
            ],
            "imports": ["src/app.py"],    # ← circular!
        },
    }

    rg = RepoGraph()

    print("=" * 60)
    print("  Commit 1 — clean graph")
    print("=" * 60)
    rg.update_commit_state("aaa111", commit1_data)
    m1 = rg.calculate_metrics()
    print(json.dumps(m1, indent=2))

    print("\n" + "=" * 60)
    print("  Commit 2 — circular dependency introduced")
    print("=" * 60)
    rg.update_commit_state("bbb222", commit2_data)
    m2 = rg.calculate_metrics()
    print(json.dumps(m2, indent=2))

    print("\n" + "=" * 60)
    print("  React Flow export (edges sample)")
    print("=" * 60)
    state = rg.export_graph_state()
    print(f"  Nodes: {len(state['nodes'])}")
    print(f"  Edges: {len(state['edges'])}")
    print()
    for edge in state["edges"]:
        flag = "  *** CYCLE VIOLATION ***" if edge["is_violation"] else ""
        print(f"  {edge['source']:<20} -> {edge['target']:<20}{flag}")

    print()
    print("  Full React Flow payload (first node):")
    print(json.dumps(state["nodes"][0], indent=4))
