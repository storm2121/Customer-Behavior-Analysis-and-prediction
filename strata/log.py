"""Structured logging that cannot leak identifiers.

Logs go to stderr so that stdout stays clean for machine-readable output. Two processors stand
between any call site and the terminal: one replaces fields whose *name* says they hold an
identifier, the other masks identifier-shaped tokens inside any string value, because a careless
f-string is the most likely way an identifier would reach a log line.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

REDACTED = "[redacted]"

IDENTIFIER_FIELDS = frozenset(
    {"client_id", "identifiant_client", "identifier", "cid", "client", "client_ids"}
)
"""Event-dict keys whose values are always replaced, whatever they contain."""

# Opaque identifiers in the extract are long hexadecimal tokens; the same conservative shape the
# privacy guard falls back to. Masking a harmless hash in a log line costs nothing.
_TOKEN = re.compile(r"(?<![0-9A-Za-z])[0-9A-Fa-f]{32,40}(?![0-9A-Za-z])")


def mask_identifiers(value: str) -> str:
    """Replace identifier-shaped tokens in free text."""
    return _TOKEN.sub(REDACTED, value)


def redact_identifiers(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """structlog processor: drop identifier fields and mask identifier-shaped tokens."""
    for key in list(event_dict):
        value = event_dict[key]
        if key in IDENTIFIER_FIELDS:
            event_dict[key] = REDACTED
        elif isinstance(value, str):
            event_dict[key] = mask_identifiers(value)
    return event_dict


def configure_logging(level: str = "INFO", *, json: bool = False) -> None:
    """Configure structlog once per process. Safe to call again (the last call wins)."""
    numeric = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            redact_identifiers,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    """A bound logger tagged with the module name."""
    logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(name)
    return logger.bind(module=name)
