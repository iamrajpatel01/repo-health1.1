
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from git_walker import GitWalker
from ast_parser import CodeParser
from graph_engine import RepoGraph



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


_ANOMALY_THRESHOLD: float = 5.0


_MAX_FILE_BYTES: int = 2 * 1024 * 1024   
_BUS_FACTOR_THRESHOLD: float = 0.80




def _read_file_at_commit(
    repo_path: Path,
    commit_hash: str,
    file_path: str,
) -> str | None:
  
    try:
        result = subprocess.run(
            ["git", "show", f"{commit_hash}:{file_path}"],
            cwd=repo_path,
            capture_output=True,
            check=True,                # raises if file was deleted (exit 128)
            timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    raw_bytes = result.stdout
    if len(raw_bytes) > _MAX_FILE_BYTES:
        logger.debug("Skipping oversized file (%d bytes): %s", len(raw_bytes), file_path)
        return None

    return raw_bytes.decode("utf-8", errors="replace")


def _get_commit_diff(repo_path: Path, commit_hash: str) -> str:
 
    try:
        result = subprocess.run(
            ["git", "diff", f"{commit_hash}^", commit_hash],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
            encoding="utf-8",
            errors="replace",
        )
        return result.stdout[:3000]
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not extract diff for %s: %s", commit_hash, exc)
        return "Diff unavailable."



def _compute_module_bus_factors(
    gross_additions: dict[str, dict[str, int]]
) -> dict[str, float]:
   
    result: dict[str, float] = {}
    for module, author_map in gross_additions.items():
        total = sum(author_map.values())
        if total == 0:
            result[module] = 10.0  # no code = no risk
            continue
        max_share = max(author_map.values()) / total
        result[module] = 1.0 if max_share >= _BUS_FACTOR_THRESHOLD else float(len(author_map))
    return result


def _build_llm_trigger(
    anomaly_detected: bool,
    prev_score: float,
    curr_score: float,
    commit_hash: str,
    repo_path: Path,
    topo_delta: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {"anomaly_detected": anomaly_detected}
    if anomaly_detected:
        drop = round(prev_score - curr_score, 2)
        payload["trigger_reason"] = (
            f"Score dropped by {drop} points "
            f"(from {prev_score} to {curr_score}) "
            f"in commit {commit_hash[:10]}"
        )
        payload["git_diff_snippet"] = _get_commit_diff(repo_path, commit_hash)
        # Topological Delta gives the LLM critical architectural context
        if topo_delta:
            payload["topological_delta"] = topo_delta
    return payload



def run_pipeline(repo_path: Path, output_path: Path, max_commits: int = 1000) -> None:
    logger.info("Initialising components for repo: %s", repo_path)

    walker = GitWalker(repo_path)
    parser = CodeParser()
    rg     = RepoGraph()

    commits      = walker.get_all_commits_data()
    total        = len(commits)
    prev_score   = 100.0
    results: list[dict[str, Any]] = []

    gross_additions: dict[str, dict[str, int]] = {}
    prev_edges: set[tuple[str, str]] = set()

    if total == 0:
        logger.warning("No commits found in repository. Exiting.")
        return

    logger.info("Found %d commit(s). Processing up to %d. Starting ingestion loop...", total, max_commits)
    pipeline_start = time.monotonic()

    for idx, (commit_hash, timestamp, author) in enumerate(commits, start=1):
        if idx > max_commits:
            logger.info("Commit cap (%d) reached. Stopping.", max_commits)
            break
        logger.info("Processing commit %d/%d  [%s]  %s", idx, total, commit_hash[:10], author)
        changed_files = walker.get_changed_files(commit_hash)
        parsed_files: dict[str, dict[str, Any]] = {}

        for cf in changed_files:
            module = cf.path.split("/")[0] if "/" in cf.path else "root"
            gross_additions.setdefault(module, {})
            gross_additions[module][author] = (
                gross_additions[module].get(author, 0) + cf.additions
            )

            content = _read_file_at_commit(repo_path, commit_hash, cf.path)
            if content is None:
                continue
            parsed_files[cf.path] = parser.parse_file(cf.path, content)
        module_bus_factors = _compute_module_bus_factors(gross_additions)
        if parsed_files:
            rg.update_commit_state(commit_hash, parsed_files, module_bus_factors)
        else:
            rg.update_commit_state(commit_hash, {}, module_bus_factors)
        curr_edges: set[tuple[str, str]] = set(rg.graph.edges())
        new_edges  = curr_edges - prev_edges
        topo_delta = ""
        if new_edges:
            samples = list(new_edges)[:5] 
            lines = [f"  {s} → {t}" for s, t in samples]
            topo_delta = (
                f"This commit introduced {len(new_edges)} new structural "
                f"dependency edge(s):\n" + "\n".join(lines)
            )
            if rg.calculate_metrics()["dependency_cycles"] > 0:
                topo_delta += "\nWARNING: one or more new edges created a circular dependency cycle."
        prev_edges = curr_edges
        metrics     = rg.calculate_metrics()
        curr_score  = metrics["overall_score"]
        score_drop  = prev_score - curr_score
        anomaly     = score_drop > _ANOMALY_THRESHOLD

        if anomaly:
            logger.warning(
                "ANOMALY DETECTED at commit %s — score dropped %.2f pts "
                "(%.2f → %.2f)",
                commit_hash[:10], score_drop, prev_score, curr_score,
            )

        llm_trigger = _build_llm_trigger(
            anomaly, prev_score, curr_score, commit_hash, repo_path, topo_delta
        )
        record: dict[str, Any] = {
            "commit_hash":       commit_hash,
            "timestamp":         timestamp,
            "author":            author,
            "health_metrics":    metrics,
            "graph_state":       rg.export_graph_state(),
            "llm_trigger_payload": llm_trigger,
        }
        results.append(record)

        prev_score = curr_score

    elapsed = time.monotonic() - pipeline_start
    logger.info(
        "Ingestion complete. %d commit(s) processed in %.1f s.",
        total, elapsed,
    )

    # ── Serialise to JSON ────────────────────────────────────────────────
    logger.info("Writing output to: %s", output_path)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    anomaly_count = sum(1 for r in results if r["llm_trigger_payload"]["anomaly_detected"])
    logger.info(
        "Done. Output: %s  |  Commits: %d  |  LLM triggers: %d",
        output_path, total, anomaly_count,
    )

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Git ingestion pipeline — produces pipeline_output.json",
    )
    p.add_argument(
        "repo",
        nargs="?",
        default=".",
        help="Path to the git repository (default: current directory).",
    )
    p.add_argument(
        "--output", "-o",
        default="pipeline_output.json",
        help="Destination JSON file (default: pipeline_output.json).",
    )
    p.add_argument(
        "--max-commits", "-n",
        type=int,
        default=1000,
        help="Maximum number of commits to process (default: 1000). "
             "Use to target a specific window in large repos.",
    )
    p.add_argument(
        "--compare",
        metavar="BRANCH",
        default=None,
        help="Run a pre-merge simulation: compare current HEAD against this branch "
             "using git worktree. Outputs a health diff JSON instead of the timeline.",
    )
    return p.parse_args(argv)


def _run_premerge_simulation(repo_path: Path, branch: str, output_path: Path) -> None:
    import tempfile, shutil
    worktree_path = Path(tempfile.mkdtemp(prefix="rh-premerge-"))
    try:
        logger.info("Pre-merge simulation: checking out '%s' via git worktree...", branch)
        subprocess.run(
            ["git", "worktree", "add", str(worktree_path), branch],
            cwd=repo_path, check=True, capture_output=True
        )

        parser = CodeParser()

        def _score_directory(path: Path) -> tuple[float, int]:
            """Parse all supported files in *path* and return (health_score, cycle_count)."""
            rg = RepoGraph()
            parsed: dict[str, Any] = {}
            for f in path.rglob("*"):
                if f.suffix.lower() in (".py", ".ts", ".tsx", ".js", ".jsx"):
                    try:
                        parsed[str(f.relative_to(path))] = parser.parse_file(
                            str(f), f.read_text(encoding="utf-8", errors="replace")
                        )
                    except Exception:  # noqa: BLE001
                        pass
            rg.update_commit_state("HEAD", parsed)
            m = rg.calculate_metrics()
            return m["overall_score"], m["dependency_cycles"]

        logger.info("Scoring HEAD (main)...")
        main_score, main_cycles = _score_directory(repo_path)
        logger.info("Scoring branch '%s'...", branch)
        branch_score, branch_cycles = _score_directory(worktree_path)

        delta = round(branch_score - main_score, 2)
        result = {
            "base_branch": "HEAD",
            "compare_branch": branch,
            "base_health_score": main_score,
            "branch_health_score": branch_score,
            "health_delta": delta,
            "base_cycles": main_cycles,
            "branch_cycles": branch_cycles,
            "verdict": "SAFE" if delta >= -3 else ("CAUTION" if delta >= -8 else "RISKY"),
            "narrative": (
                f"Merging '{branch}' would shift health from {main_score} → {branch_score} "
                f"({'+' if delta >= 0 else ''}{delta} pts) and change dependency cycles "
                f"from {main_cycles} to {branch_cycles}."
            ),
        }
        with open(output_path, "w") as fh:
            json.dump(result, fh, indent=2)
        logger.info("Pre-merge report written to %s", output_path)
        logger.info("Verdict: %s | Health delta: %+.2f", result['verdict'], delta)

    finally:
        subprocess.run(
            ["git", "worktree", "remove", str(worktree_path), "--force"],
            cwd=repo_path, capture_output=True
        )
        shutil.rmtree(worktree_path, ignore_errors=True)


if __name__ == "__main__":
    args      = _parse_args()
    repo_path = Path(args.repo).resolve()
    out_path  = Path(args.output).resolve()

    if not (repo_path / ".git").exists():
        logger.error("'%s' is not a git repository.", repo_path)
        sys.exit(1)

    if args.compare:
        _run_premerge_simulation(repo_path, args.compare, out_path)
    else:
        run_pipeline(repo_path, out_path, max_commits=args.max_commits)
