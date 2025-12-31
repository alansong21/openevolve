"""
Lightweight bridge for the optional LangChain code-context agent.

The goal is to keep the core evolution loop decoupled: if the agent or its
dependencies are missing, we simply return an empty context bundle and let the
loop proceed unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    # Prefer the in-tree agent to avoid PYTHONPATH issues.
    from openevolve.internal_agent import agent as _agent  # type: ignore
except Exception as exc:  # noqa: BLE001
    _agent = None  # type: ignore
    _import_error = exc


def _coerce_artifacts(artifacts: Dict[str, Any] | None, limit: int = 4000) -> str:
    """Render artifacts to a short plaintext blob for the context query."""
    if not artifacts:
        return ""
    parts = []
    for name, payload in artifacts.items():
        if isinstance(payload, bytes):
            try:
                payload = payload.decode("utf-8", errors="ignore")
            except Exception:
                payload = ""
        elif not isinstance(payload, str):
            payload = str(payload)
        parts.append(f"[{name}]\n{payload}")
    blob = "\n\n".join(parts)
    return blob[:limit]


def fetch_context_from_agent(
    task_description: str,
    artifacts: Dict[str, Any] | None = None,
    token_budget: int = 1200,
) -> str:
    """
    Ask the external LangChain agent for a compact codebase context bundle.

    Returns an empty string if the agent is unavailable or an error occurs.
    """
    if _agent is None:
        if _import_error:
            logger.debug("Context agent unavailable: %s", _import_error)
        return ""

    logger.debug("Invoking context agent (budget=%d)", token_budget)
    artifact_text = _coerce_artifacts(artifacts)
    query = (
        "Summarize only the ChampSim/OpenEvolve APIs and source snippets that are relevant "
        "to the current task. Prefer prefetcher APIs, address slices, access_type enums, "
        "and L2-related code. Keep the output concise.\n"
        f"Task: {task_description}\n"
        f"Recent artifacts:\n{artifact_text}"
    )

    try:
        result = _agent.invoke({"messages": [{"role": "user", "content": query}]})
        messages = result.get("messages", []) if isinstance(result, dict) else []
        content = messages[-1].content if messages else ""
    except Exception as exc:  # noqa: BLE001
        logger.debug("Context agent invocation failed: %s", exc)
        return ""

    if not content:
        logger.debug("Context agent returned empty content")
        return ""

    trimmed = content[:token_budget]
    logger.debug("Context agent returned %d chars (trimmed to %d)", len(content), len(trimmed))
    return trimmed
