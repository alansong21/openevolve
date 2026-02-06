"""
Programmatic, in-tree context fetcher for hardcoded ChampSim files.

This keeps a LangChain-based agent available, but defaults to the deterministic
programmatic fetcher to avoid external dependencies.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

# Optional LangChain agent dependencies.
try:
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI

    _LANGCHAIN_AVAILABLE = True
except Exception:  # noqa: BLE001
    _LANGCHAIN_AVAILABLE = False

# Resolve project root (two levels up from this file: openevolve/openevolve/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

HARDCODED_FILES = [
    "ChampSim/inc/address.h",
    "ChampSim/inc/champsim.h",
    "ChampSim/inc/modules.h",
    "ChampSim/inc/cache.h",
]


def _read_hardcoded_files() -> str:
    sections: List[str] = []
    for rel_path in HARDCODED_FILES:
        path = (PROJECT_ROOT / rel_path).resolve()
        if not path.is_file():
            sections.append(f"[{rel_path}]\n<missing>")
            continue
        try:
            content = path.read_text()
        except Exception as exc:  # noqa: BLE001
            sections.append(f"[{rel_path}]\n<read error: {exc}>")
            continue
        sections.append(f"[{rel_path}]\n{content}")
    return "\n\n".join(sections)


class _ProgrammaticAgent:
    """Minimal stand-in that matches the LangChain agent invoke() API."""

    def invoke(self, _: Dict[str, Any]) -> Dict[str, Any]:
        content = _read_hardcoded_files()
        return {"messages": [{"content": content}]}

def _build_langchain_agent() -> Any | None:
    if not _LANGCHAIN_AVAILABLE:
        return None

    # Named roots used for list/read tools. Include broad aliases so callers do not
    # have to guess the exact subpath.
    ROOTS = {
        "repo": PROJECT_ROOT,
        "champsim": (PROJECT_ROOT / "ChampSim").resolve(),
        "champsim/inc": (PROJECT_ROOT / "ChampSim" / "inc").resolve(),
        "champsim/src": (PROJECT_ROOT / "ChampSim" / "src").resolve(),
        "openevolve": (PROJECT_ROOT / "openevolve").resolve(),
        "components": (PROJECT_ROOT / "openevolve-components").resolve(),
        "prefetchers": (PROJECT_ROOT / "prefetchers").resolve(),
    }

    def _resolve_root(alias: str | None = None) -> Path:
        """Resolve a root alias, falling back to the repo root if unknown."""
        if alias is None:
            raise ValueError("No root alias provided.")
        try:
            return ROOTS[alias]
        except KeyError:
            return ROOTS["repo"]

    def _find_roots_for_path(rel_path: str, must_be_dir: bool) -> list[tuple[str, Path]]:
        matches: list[tuple[str, Path]] = []
        for alias, root in ROOTS.items():
            candidate = (root / rel_path).resolve()
            if root not in candidate.parents and candidate != root:
                continue
            if must_be_dir and candidate.is_dir():
                matches.append((alias, candidate))
            elif not must_be_dir and candidate.is_file():
                matches.append((alias, candidate))
        return matches

    @tool
    def list_files(rel_path: str = ".", root_alias: str | None = None) -> str:
        """List files. If root_alias is omitted or unknown, search all known roots."""
        if rel_path in ("", ".") and root_alias is None:
            sections = []
            for alias, root in ROOTS.items():
                try:
                    items = [p.relative_to(root).as_posix() for p in root.iterdir()]
                    sections.append(f"[{alias}]\n" + "\n".join(sorted(items)))
                except FileNotFoundError:
                    sections.append(f"[{alias}]\n(root missing)")
            return "\n\n".join(sections)

        if root_alias:
            root = _resolve_root(root_alias)
            base = (root / rel_path).resolve()
            if root in base.parents or base == root:
                if not base.exists():
                    return "Path not found."
                if not base.is_dir():
                    return "Not a directory."
                items = [p.relative_to(root).as_posix() for p in base.iterdir()]
                return "\n".join(sorted(items))
            # Fall back to searching all roots if outside

        matches = _find_roots_for_path(rel_path, must_be_dir=True)
        if not matches:
            return "Path not found in any root."
        if len(matches) > 1:
            choices = ", ".join(a for a, _ in matches)
            return f"Ambiguous path. Found in: {choices}. Specify root_alias."
        alias, base = matches[0]
        items = [p.relative_to(ROOTS[alias]).as_posix() for p in base.iterdir()]
        return "\n".join(sorted(items))

    @tool
    def read_file(rel_path: str, root_alias: str | None = None) -> str:
        """Read a file. If root_alias is omitted or unknown, search all known roots."""
        if root_alias:
            root = _resolve_root(root_alias)
            target = (root / rel_path).resolve()
            if root in target.parents or target == root:
                if not target.is_file():
                    return "Not a file."
                return target.read_text()
            # Fall back to searching all roots

        matches = _find_roots_for_path(rel_path, must_be_dir=False)
        if not matches:
            return "File not found in any root."
        if len(matches) > 1:
            choices = ", ".join(a for a, _ in matches)
            return f"Ambiguous file. Found in: {choices}. Specify root_alias."
        alias, target = matches[0]
        return target.read_text()

    tools = [list_files, read_file]

    system_prompt = (
        "You are a helpful coding assistant. "
        "Use list_files and read_file to gather only the most relevant ChampSim/OpenEvolve context. "
        "You may explore all known roots without specifying a root_alias unless needed."
    )

    llm = ChatOpenAI(model="gpt-4o-mini")
    return create_agent(model=llm, tools=tools, system_prompt=system_prompt)


def _select_agent() -> Any:
    mode = os.getenv("OPENEvolve_CONTEXT_AGENT", "programmatic").lower()
    if mode == "langchain":
        langchain_agent = _build_langchain_agent()
        if langchain_agent is not None:
            return langchain_agent
    return _ProgrammaticAgent()


agent = _select_agent()
