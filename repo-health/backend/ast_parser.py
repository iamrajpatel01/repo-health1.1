"""
ast_parser.py
-------------
AST Parsing Engine — Part 2 of the Git ingestion pipeline.

Responsibilities:
  - Accept a single file's content as a string
  - Detect language from the file extension (.py, .js, .ts)
  - Parse with Tree-Sitter (v0.25 API) and extract:
      * functions / classes  →  name, kind, start_line, cyclomatic_complexity
      * imports              →  the module / path being imported
  - Return a clean dict ready for downstream graph / API consumption

GUARDRAIL: No NetworkX, no cross-file resolution, no graph logic here.
           This module is purely single-file → structured dict.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy language registry
# ---------------------------------------------------------------------------
# We import language bindings inside a try/except so the module loads even
# when a particular grammar wheel is not installed.  Missing grammars are
# logged once and gracefully skipped at parse time.

def _load_languages() -> dict[str, Any]:
    """
    Attempt to import every supported Tree-Sitter grammar.
    Returns a mapping of extension → tree_sitter.Language (or None).
    """
    from tree_sitter import Language  # hard dependency — must be installed

    registry: dict[str, Any] = {}

    # Python
    try:
        import tree_sitter_python as _tspy
        py_lang = Language(_tspy.language())
        registry[".py"] = py_lang
        logger.debug("tree-sitter: Python grammar loaded.")
    except ImportError:
        logger.warning("tree-sitter-python not installed — .py files will be skipped.")
        registry[".py"] = None

    # JavaScript / TypeScript share the JS grammar for now;
    # tree-sitter-javascript handles both .js and .ts at the token level.
    try:
        import tree_sitter_javascript as _tsjs
        js_lang = Language(_tsjs.language())
        registry[".js"]  = js_lang
        registry[".ts"]  = js_lang
        registry[".jsx"] = js_lang
        registry[".tsx"] = js_lang
        logger.debug("tree-sitter: JavaScript grammar loaded (covers .js/.ts/.jsx/.tsx).")
    except ImportError:
        logger.warning(
            "tree-sitter-javascript not installed — .js/.ts files will be skipped."
        )
        for ext in (".js", ".ts", ".jsx", ".tsx"):
            registry[ext] = None

    return registry


# Module-level cache — populated on first use
_LANG_CACHE: dict[str, Any] | None = None


def _get_language(ext: str) -> Any | None:
    """Return the Language object for *ext*, or None if unavailable."""
    global _LANG_CACHE
    if _LANG_CACHE is None:
        try:
            _LANG_CACHE = _load_languages()
        except ImportError:
            logger.error(
                "tree-sitter core library not installed. "
                "Run: pip install tree-sitter tree-sitter-python tree-sitter-javascript"
            )
            _LANG_CACHE = {}
    return _LANG_CACHE.get(ext)


# ---------------------------------------------------------------------------
# Cyclomatic-complexity node types per language
# ---------------------------------------------------------------------------
# Each branching keyword increments the count by 1.
# (baseline is 1; we start counting at 0 and add 1 at the end)

_COMPLEXITY_NODES: dict[str, frozenset[str]] = {
    ".py": frozenset({
        "if_statement",
        "elif_clause",
        "for_statement",
        "while_statement",
        "except_clause",
        "with_statement",
        "conditional_expression",   # inline ternary: x if cond else y
        "boolean_operator",         # and / or  (each adds a branch)
    }),
    ".js": frozenset({
        "if_statement",
        "else_clause",
        "for_statement",
        "for_in_statement",
        "while_statement",
        "do_statement",
        "catch_clause",
        "ternary_expression",
        "logical_expression",       # && / ||
        "switch_case",
    }),
}

# .ts / .jsx / .tsx reuse the JS set
for _ext in (".ts", ".jsx", ".tsx"):
    _COMPLEXITY_NODES[_ext] = _COMPLEXITY_NODES[".js"]


# ---------------------------------------------------------------------------
# Node-type tables for definitions & imports
# ---------------------------------------------------------------------------

# Maps extension → (definition_node_types, import_node_types)
_DEFINITION_TYPES: dict[str, frozenset[str]] = {
    ".py":  frozenset({"function_definition", "class_definition", "async_function_definition"}),
    ".js":  frozenset({
        "function_declaration", "function_expression", "arrow_function",
        "class_declaration", "method_definition", "generator_function_declaration",
    }),
}
for _ext in (".ts", ".jsx", ".tsx"):
    _DEFINITION_TYPES[_ext] = _DEFINITION_TYPES[".js"]

_IMPORT_TYPES: dict[str, frozenset[str]] = {
    ".py":  frozenset({"import_statement", "import_from_statement"}),
    ".js":  frozenset({"import_statement", "import_declaration"}),
}
for _ext in (".ts", ".jsx", ".tsx"):
    _IMPORT_TYPES[_ext] = _IMPORT_TYPES[".js"]


# ---------------------------------------------------------------------------
# Internal tree-walking helpers
# ---------------------------------------------------------------------------

def _iter_descendants(node: Any) -> "list[Any]":
    """
    Return every descendant node (breadth-first) under *node*, including
    *node* itself.  Uses the node.children list from tree-sitter.
    """
    result: list[Any] = []
    stack = [node]
    while stack:
        current = stack.pop()
        result.append(current)
        stack.extend(current.children)
    return result


def _cyclomatic_complexity(node: Any, complexity_types: frozenset[str]) -> int:
    """
    Compute McCabe cyclomatic complexity for a single definition node.

    Algorithm: baseline of 1 + 1 for every branching descendant node
    whose type is listed in *complexity_types*.
    """
    count = 1  # baseline
    for desc in _iter_descendants(node):
        if desc.type in complexity_types:
            count += 1
    return count


def _node_name(node: Any) -> str:
    """
    Extract the name identifier from a definition node.

    Tree-Sitter exposes named fields via child_by_field_name().
    Falls back to scanning immediate children for an 'identifier' node.
    """
    # Fast path: named field 'name' exists on most definition nodes
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return (name_node.text or b"<anonymous>").decode("utf-8", errors="replace")

    # Slow path: first identifier child
    for child in node.children:
        if child.type == "identifier":
            return (child.text or b"<anonymous>").decode("utf-8", errors="replace")

    return "<anonymous>"


def _extract_python_import(node: Any) -> list[str]:
    """
    Return a list of module strings from a Python import node.

    Handles:
      import os                   → ["os"]
      import os, sys              → ["os", "sys"]
      from pathlib import Path    → ["pathlib"]
      from . import utils         → ["."]
    """
    modules: list[str] = []

    if node.type == "import_statement":
        # Children: import <dotted_name | aliased_import> [, ...]
        for child in node.children:
            if child.type in ("dotted_name", "aliased_import"):
                # dotted_name has the module; aliased_import wraps it
                target = child.child_by_field_name("name") or child
                text = (target.text or b"").decode("utf-8", errors="replace").strip()
                if text:
                    modules.append(text)

    elif node.type == "import_from_statement":
        # 'from <module> import ...'  — we only care about the source module
        module_node = node.child_by_field_name("module_name")
        if module_node:
            text = (module_node.text or b"").decode("utf-8", errors="replace").strip()
            if text:
                modules.append(text)
        else:
            # Relative import: 'from . import x'  — emit "." as the source
            for child in node.children:
                if child.type == "relative_import":
                    modules.append(".")
                    break

    return modules


def _extract_js_import(node: Any) -> list[str]:
    """
    Return module source strings from a JS/TS ESM import node.

    Handles:
      import foo from './foo'           → ["./foo"]
      import { bar } from '../bar'      → ["../bar"]
      import * as x from 'lodash'       → ["lodash"]

    CommonJS ``require()`` calls are handled separately by
    :func:`_extract_require`.
    """
    modules: list[str] = []

    # import_statement / import_declaration both have a 'source' field
    source_node = node.child_by_field_name("source")
    if source_node:
        raw = (source_node.text or b"").decode("utf-8", errors="replace").strip()
        # Strip surrounding quotes
        if raw and raw[0] in ('"', "'", "`"):
            raw = raw[1:-1]
        if raw:
            modules.append(raw)

    return modules


def _extract_require(node: Any) -> list[str]:
    """
    Extract the module string from a CommonJS ``require('...')`` call.

    Tree-Sitter represents ``require('express')`` as::

        call_expression
          function: identifier        [text = 'require']
          arguments: arguments
            string                    [text = "'express'"]

    Returns a one-element list with the unquoted module string, or an
    empty list if the call_expression is not a ``require`` call or if
    the argument is not a plain string literal.
    """
    # The callee must be exactly the identifier 'require'
    callee = node.child_by_field_name("function")
    if callee is None or callee.type != "identifier":
        return []
    callee_name = (callee.text or b"").decode("utf-8", errors="replace").strip()
    if callee_name != "require":
        return []

    # Dig into the arguments list for the first string literal
    args_node = node.child_by_field_name("arguments")
    if args_node is None:
        return []

    for arg in args_node.children:
        if arg.type == "string":
            raw = (arg.text or b"").decode("utf-8", errors="replace").strip()
            # Strip surrounding quotes (' " `)
            if raw and raw[0] in ('"', "'", "`"):
                raw = raw[1:-1]
            if raw:
                return [raw]

    return []


# ---------------------------------------------------------------------------
# CodeParser
# ---------------------------------------------------------------------------

class CodeParser:
    """
    Parse a single source file and extract structural metadata.

    This class is stateless — each call to :meth:`parse_file` is independent.
    Instantiate once and reuse across thousands of files efficiently.

    The returned dictionary has the shape::

        {
            "functions": [
                {
                    "name":        str,   # identifier (or "<anonymous>")
                    "kind":        str,   # "function" | "class" | "method"
                    "start_line":  int,   # 1-indexed
                    "end_line":    int,   # 1-indexed
                    "complexity":  int,   # McCabe cyclomatic complexity ≥ 1
                },
                ...
            ],
            "imports": [
                str,   # module / file path being imported
                ...
            ],
        }
    """

    def parse_file(self, file_path: str, file_content: str) -> dict[str, Any]:
        """
        Parse *file_content* and return extracted structural data.

        Parameters
        ----------
        file_path : str
            Used only to determine the language from the file extension.
            The file does **not** have to exist on disk.
        file_content : str
            Full source text of the file.

        Returns
        -------
        dict
            ``{"functions": [...], "imports": [...]}``
            Returns an empty structure for unsupported extensions or on
            parse error.  Never raises.
        """
        empty: dict[str, Any] = {"functions": [], "imports": []}

        ext = Path(file_path).suffix.lower()
        lang = _get_language(ext)

        if lang is None:
            if ext:
                logger.debug("Unsupported extension '%s' — skipping %s", ext, file_path)
            return empty

        # ── Parse ───────────────────────────────────────────────────────
        try:
            from tree_sitter import Parser as TSParser
            parser = TSParser(lang)
            tree = parser.parse(file_content.encode("utf-8", errors="replace"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tree-Sitter parse error for %s: %s", file_path, exc)
            return empty

        root = tree.root_node

        # ── Walk the tree once, collecting definitions and imports ───────
        def_types    = _DEFINITION_TYPES.get(ext, frozenset())
        import_types = _IMPORT_TYPES.get(ext, frozenset())
        cc_types     = _COMPLEXITY_NODES.get(ext, frozenset())

        functions: list[dict[str, Any]] = []
        imports:   list[str]            = []

        self._walk(
            node         = root,
            ext          = ext,
            def_types    = def_types,
            import_types = import_types,
            cc_types     = cc_types,
            functions    = functions,
            imports      = imports,
        )

        return {"functions": functions, "imports": sorted(set(imports))}

    # ------------------------------------------------------------------
    # Private — tree walk
    # ------------------------------------------------------------------

    def _walk(
        self,
        node: Any,
        ext: str,
        def_types: frozenset[str],
        import_types: frozenset[str],
        cc_types: frozenset[str],
        functions: list[dict[str, Any]],
        imports: list[str],
    ) -> None:
        """
        Recursively walk the CST rooted at *node*.

        Definitions are extracted with their cyclomatic complexity.
        Nested definitions (methods inside classes) are also captured.
        """
        for child in node.children:
            ntype = child.type

            # ── Definition node ─────────────────────────────────────────
            if ntype in def_types:
                kind = self._classify_kind(ntype)
                name = _node_name(child)
                complexity = _cyclomatic_complexity(child, cc_types)

                functions.append({
                    "name":       name,
                    "kind":       kind,
                    "start_line": child.start_point[0] + 1,  # 0-indexed → 1-indexed
                    "end_line":   child.end_point[0]   + 1,
                    "complexity": complexity,
                })

                # Recurse into the body to capture nested defs (methods etc.)
                self._walk(child, ext, def_types, import_types, cc_types, functions, imports)

            # ── ESM import node ──────────────────────────────────────────
            elif ntype in import_types:
                if ext == ".py":
                    imports.extend(_extract_python_import(child))
                else:
                    imports.extend(_extract_js_import(child))

                # No need to recurse into import nodes

            # ── CommonJS require() call  (JS/TS only) ────────────────────
            elif ntype == "call_expression" and ext not in (".py",):
                found = _extract_require(child)
                if found:
                    imports.extend(found)
                    # No need to recurse into this call_expression
                else:
                    # Not a require() — still walk in case it contains
                    # nested definitions or other require() calls
                    self._walk(child, ext, def_types, import_types, cc_types, functions, imports)

            else:
                # Keep walking — definitions can appear in any block scope
                self._walk(child, ext, def_types, import_types, cc_types, functions, imports)

    @staticmethod
    def _classify_kind(node_type: str) -> str:
        """Map a raw node-type string to a human-readable kind label."""
        if "class" in node_type:
            return "class"
        if "method" in node_type:
            return "method"
        return "function"


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import io
    import json
    import sys

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = CodeParser()

    # ── Python sample ────────────────────────────────────────────────────
    py_source = '''\
import os
import sys
from pathlib import Path
from collections import defaultdict

class DataProcessor:
    """Processes data from multiple sources."""

    def __init__(self, config: dict):
        self.config = config

    def load(self, path: str) -> list:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        data = []
        for line in open(path):
            if line.strip():
                data.append(line)
        return data

    def transform(self, data: list) -> list:
        result = []
        for item in data:
            while item.endswith("\\n"):
                item = item[:-1]
            try:
                result.append(int(item))
            except ValueError:
                pass
        return result


def top_level_util(x: int) -> int:
    """Simple branching utility."""
    if x > 0:
        return x * 2
    elif x < 0:
        return -x
    return 0
'''

    # ── JavaScript sample ────────────────────────────────────────────────
    js_source = '''\
import express from 'express';
import { readFile } from 'fs/promises';
import path from 'path';

// CommonJS-style require() — must also appear in imports
const mongoose = require('mongoose');
const { Router } = require('express');
const config = require('./config/settings');

class ApiServer {
    constructor(port) {
        this.port = port;
    }

    start() {
        const app = express();
        if (!this.port) {
            throw new Error('No port configured');
        }
        for (const route of this.routes) {
            app.use(route);
        }
        app.listen(this.port);
    }
}

async function fetchData(url) {
    try {
        const res = await fetch(url);
        if (!res.ok) {
            throw new Error(res.statusText);
        }
        return await res.json();
    } catch (err) {
        console.error(err);
        return null;
    }
}
'''

    for label, source, path in [
        ("Python", py_source, "processor.py"),
        ("JavaScript", js_source, "server.js"),
    ]:
        print("=" * 60)
        print(f"  {label}  →  {path}")
        print("=" * 60)
        result = parser.parse_file(path, source)

        print(f"\n  Imports ({len(result['imports'])}):")
        for imp in result["imports"]:
            print(f"    • {imp}")

        print(f"\n  Definitions ({len(result['functions'])}):")
        for fn in result["functions"]:
            cc_bar = "█" * min(fn["complexity"], 10)
            print(
                f"    [{fn['kind']:<8}]  {fn['name']:<25}"
                f"  lines {fn['start_line']:>3}–{fn['end_line']:<3}"
                f"  CC={fn['complexity']}  {cc_bar}"
            )
        print()

    # ── Unsupported extension ────────────────────────────────────────────
    print("Unsupported extension test (.png):")
    result = parser.parse_file("image.png", "binary data")
    print(" ", result)
