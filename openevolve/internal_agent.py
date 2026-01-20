"""
Lightweight, in-tree LangChain agent for fetching code/context snippets.

This mirrors the external agent/agent.py but lives inside the package so it
does not depend on PYTHONPATH tweaks.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import threading
import time

from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Resolve project root (two levels up from this file: openevolve/openevolve/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

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


LOG_MAX_CHARS = int(os.environ.get("INTERNAL_AGENT_LOG_MAX_CHARS", 8000))


def _resolve_agent_log_path() -> Path:
    run_id = os.environ.get("OPENEVOLVE_RUN_ID", "").strip()
    if run_id:
        return (
            PROJECT_ROOT
            / "openevolve-components"
            / "openevolve_output"
            / "runs"
            / run_id
            / "openevolve"
            / "internal_agent.jsonl"
        )
    return (
        PROJECT_ROOT
        / "openevolve-components"
        / "openevolve_output"
        / "logs"
        / "internal_agent.jsonl"
    )


def _truncate(payload: str) -> str:
    if len(payload) <= LOG_MAX_CHARS:
        return payload
    return payload[:LOG_MAX_CHARS] + "...(truncated)"


class AgentLogHandler(BaseCallbackHandler):
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.lock = threading.Lock()

    def _write(self, event: str, payload: dict) -> None:
        record = {
            "timestamp": time.time(),
            "event": event,
            "payload": payload,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=True, default=str)
        with self.lock:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        safe_prompts = [_truncate(p) for p in prompts]
        self._write(
            "llm_start",
            {"serialized": serialized, "prompts": safe_prompts, "kwargs": kwargs},
        )

    def on_llm_end(self, response, **kwargs) -> None:
        content = ""
        try:
            content = response.generations[0][0].text or ""
        except Exception:
            content = str(response)
        self._write(
            "llm_end",
            {"response": _truncate(content), "kwargs": kwargs},
        )

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        self._write(
            "tool_start",
            {"serialized": serialized, "input": _truncate(str(input_str)), "kwargs": kwargs},
        )

    def on_tool_end(self, output, **kwargs) -> None:
        self._write(
            "tool_end",
            {"output": _truncate(str(output)), "kwargs": kwargs},
        )

    def on_agent_action(self, action, **kwargs) -> None:
        self._write("agent_action", {"action": str(action), "kwargs": kwargs})

    def on_agent_finish(self, finish, **kwargs) -> None:
        self._write("agent_finish", {"finish": str(finish), "kwargs": kwargs})


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

SYSTEM_PROMPT = (
    "You are a helpful coding assistant. "
    "Start every response with a short 'Known pitfalls' list. "
    "Include this pitfall verbatim: "
    "'Do NOT use std::unordered_map or std::unordered_set with champsim::address, "
    "champsim::block_number, or champsim::address_slice<> keys unless you define a custom "
    "hash; prefer converting to uint64_t keys.' "
    "Use list_files and read_file to gather only the most relevant ChampSim/OpenEvolve context. "
    "You may explore all known roots without specifying a root_alias unless needed."
)

agent_log_handler = AgentLogHandler(_resolve_agent_log_path())
llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[agent_log_handler])
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)
