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


def _coerce_artifacts(artifacts: Dict[str, Any] | None, limit: int | None = None) -> str:
    """Render artifacts to a plaintext blob for the context query."""
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
    if limit is None:
        return blob
    return blob[:limit]


def _approx_token_count(text: str) -> int:
    # Simple approximation for logging/observability only.
    return max(1, (len(text) + 3) // 4) if text else 0


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

    logger.info(
        "Invoking context agent (token_budget=%d is informational only; no truncation is currently applied to returned context)",
        token_budget,
    )
    artifact_text = _coerce_artifacts(artifacts)
    combined_text = f"{task_description}\n\n{artifact_text}".lower()
    is_cbp_ng = any(
        marker in combined_text
        for marker in (
            "cbp-ng",
            "workflows/cbp_ng",
            "openevolve_predictor.hpp",
            "initial_program.hpp",
            "harcom",
        )
    )
    if is_cbp_ng:
        bundle_name = "cbp_ng"
        file_instructions = (
            "Specifically, read and return the full contents of the following CBP-NG/HARCOM files: "
            "** 1) cbp-ng/harcom.hpp 2) cbp-ng/cbp.hpp 3) cbp-ng/README.md 4) cbp-ng/docs/tutorial.md "
            "5) cbp-ng/predictors/tutorial/tutorial_00.hpp 6) cbp-ng/predictors/tutorial/tutorial_01.hpp "
            "7) cbp-ng/predictors/tutorial/tutorial_02.hpp 8) cbp-ng/predictors/tutorial/tutorial_03.hpp "
            "9) cbp-ng/predictors/tutorial/tutorial_04.hpp 10) cbp-ng/predictors/common.hpp "
            "11) cbp-ng/predictors/bimodal.hpp 12) cbp-ng/predictors/gshare.hpp "
            "13) cbp-ng/predictors/perceptron.hpp 14) cbp-ng/predictors/tage.hpp **."
        )
    else:
        bundle_name = "champsim"
        file_instructions = (
            "Specifically, read and return the full contents of the following ChampSim files: "
            "** 1) ChampSim/inc/address.h 2) ChampSim/inc/champsim.h 3) ChampSim/inc/modules.h 4) ChampSim/inc/cache.h **."
        )
    query = (
        "Read the artifacts below, and find the information needed to resolve the errors in the artifacts by reading the relevant files from the codebase."
        f"{file_instructions}"
        f"\nTask description:\n{task_description}\n"
        f"Recent artifacts:\n{artifact_text}"
    )
    logger.info(
        "Context agent query prepared for bundle '%s': query=%d chars (~%d tokens), artifacts=%d chars (~%d tokens)",
        bundle_name,
        len(query),
        _approx_token_count(query),
        len(artifact_text),
        _approx_token_count(artifact_text),
    )

    try:
        result = _agent.invoke({"messages": [{"role": "user", "content": query}]})
        messages = result.get("messages", []) if isinstance(result, dict) else []
        if not messages:
            content = ""
        else:
            last = messages[-1]
            if isinstance(last, dict):
                content = last.get("content", "")
            else:
                content = getattr(last, "content", "")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Context agent invocation failed: %s", exc)
        return ""

    if not content:
        logger.debug("Context agent returned empty content")
        return ""

    logger.info(
        "Context agent returned non-empty bundle '%s': %d chars (~%d tokens); no trimming applied",
        bundle_name,
        len(content),
        _approx_token_count(content),
    )
    return content
