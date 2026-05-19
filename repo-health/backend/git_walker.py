"""
git_walker.py
-------------
High-performance Git ingestion engine — traverse and filter layer.

Responsibilities:
  - Walk all commits in chronological order (oldest → newest)
  - For each commit, return only the *text/code* files that changed
  - Binary files (those whose --numstat lines show '-' for adds/dels) are
    silently dropped before any downstream processing sees them.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChangedFile:
    """Represents a single text file that was touched in a commit."""
    path: str
    additions: int       # lines added  (always >= 0 for text files)
    deletions: int       # lines deleted (always >= 0 for text files)


@dataclass
class CommitInfo:
    """Lightweight commit record with its filtered changed-file list."""
    hash: str
    timestamp: str       # ISO 8601 author date  (e.g. 2024-03-15T10:22:01+05:30)
    author: str          # Author e-mail address
    changed_files: list[ChangedFile] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core walker
# ---------------------------------------------------------------------------

class GitWalker:
    """
    Traverse a Git repository and expose its history as structured data.

    Parameters
    ----------
    repo_path : str | Path
        Absolute (or relative) path to the root of a Git repository.
        The directory must contain a `.git` folder.
    """

    def __init__(self, repo_path: str | Path) -> None:
        self.repo_path = Path(repo_path).resolve()
        if not (self.repo_path / ".git").exists():
            raise ValueError(
                f"'{self.repo_path}' does not appear to be a Git repository "
                "(no .git directory found)."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(self, *args: str) -> str:
        """
        Run a git sub-command inside the repository and return stdout.

        Raises
        ------
        subprocess.CalledProcessError
            If git exits with a non-zero return code.
        """
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace",   # guard against non-UTF-8 file names
        )
        return result.stdout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_all_commits_data(self) -> list[tuple[str, str, str]]:
        """
        Return every commit's metadata reachable from HEAD, oldest first.

        Uses a custom ``git log`` format string to collect all three
        fields in a single subprocess call::

            %H  — full SHA-1 hash
            %aI — author date, strict ISO 8601 with timezone offset
            %ae — author e-mail address

        Fields are pipe-delimited (``|``) to avoid collisions with
        whitespace that may appear in e-mail addresses.

        Returns
        -------
        list[tuple[str, str, str]]
            Ordered list of ``(hash, iso_timestamp, author_email)`` tuples,
            oldest commit first.
        """
        raw = self._run("log", "--reverse", "--format=%H|%aI|%ae")
        commits: list[tuple[str, str, str]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 3:
                commits.append((parts[0], parts[1], parts[2]))
        return commits

    def get_changed_files(self, commit_hash: str) -> list[ChangedFile]:
        """
        Return the text/code files modified in *commit_hash*.

        Runs ``git diff-tree --no-commit-id --numstat -r <hash>`` and
        parses each output line, which has the format::

            <additions>\\t<deletions>\\t<path>

        GUARDRAIL — Binary file filtering
        ----------------------------------
        When git cannot diff a binary blob it emits ``-`` for both the
        additions and deletions columns.  Those entries are **silently
        excluded** so only human-readable source files flow downstream.

        Parameters
        ----------
        commit_hash : str
            A full or abbreviated commit SHA-1.

        Returns
        -------
        list[ChangedFile]
            Text files touched by this commit, with per-file line stats.
        """
        raw = self._run(
            "diff-tree", "--no-commit-id", "--numstat", "-r", commit_hash
        )

        changed: list[ChangedFile] = []

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t", maxsplit=2)
            if len(parts) != 3:
                # Malformed line — skip defensively
                continue

            additions_raw, deletions_raw, path = parts

            # ── CRITICAL GUARDRAIL ──────────────────────────────────────
            # Binary files are reported with '-' in both stat columns.
            # Filter them out unconditionally.
            # ────────────────────────────────────────────────────────────
            if additions_raw == "-" and deletions_raw == "-":
                continue  # binary file → drop

            try:
                additions = int(additions_raw)
                deletions = int(deletions_raw)
            except ValueError:
                # Unexpected format — skip rather than crash
                continue

            changed.append(ChangedFile(path=path, additions=additions, deletions=deletions))

        return changed

    # ------------------------------------------------------------------
    # Convenience iterator
    # ------------------------------------------------------------------

    def iter_commits(self) -> Iterator[CommitInfo]:
        """
        Yield :class:`CommitInfo` for every commit, oldest first.

        Each object carries the commit hash, ISO timestamp, author e-mail,
        and the pre-filtered list of changed text files — ready for
        downstream ingestion into the FastAPI / JSON layer.
        """
        for commit_hash, timestamp, author in self.get_all_commits_data():
            files = self.get_changed_files(commit_hash)
            yield CommitInfo(
                hash=commit_hash,
                timestamp=timestamp,
                author=author,
                changed_files=files,
            )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import io

    # Force UTF-8 output so non-ASCII chars render on Windows terminals
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    # Default: run against the genesis repo itself.
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent

    print(f"Repository : {repo}")
    print("-" * 60)

    walker = GitWalker(repo)

    commits_data = walker.get_all_commits_data()
    print(f"Total commits found : {len(commits_data)}")
    print()

    # Show the first 5 and last 5 entries
    sample = commits_data[:5] + ([("..", "..", "..")] if len(commits_data) > 10 else []) + commits_data[-5:]
    print(f"{'HASH':<12}  {'TIMESTAMP':<35}  AUTHOR")
    print("-" * 80)
    for h, ts, auth in sample:
        print(f"  {h[:10]}  {ts:<35}  {auth}")
    print()

    # Inspect the most recent commit in detail
    if commits_data:
        latest_hash, latest_ts, latest_author = commits_data[-1]
        files = walker.get_changed_files(latest_hash)
        print(f"Latest commit  : {latest_hash}")
        print(f"Timestamp      : {latest_ts}")
        print(f"Author         : {latest_author}")
        print(f"Changed files  : {len(files)} text file(s)")
        for f in files:
            print(f"  +{f.additions:<6} -{f.deletions:<6} {f.path}")

    print()
    print("Full walk stats (via iter_commits):")
    total_files = 0
    for info in walker.iter_commits():
        total_files += len(info.changed_files)
    print(f"  Total text-file change events across all commits: {total_files}")
