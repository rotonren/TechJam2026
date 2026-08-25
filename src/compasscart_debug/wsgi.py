"""Gunicorn entry point for the CompassCart debug WSGI application."""

from __future__ import annotations

from .http import (
    CSP,
    STATIC_ALLOWLIST,
    STATIC_MAP,
    DebugApplication,
    DebugWSGIApplication,
    WSGIApplication,
    check_bearer_token,
    create_application,
    parse_content_length,
    parse_json_body,
    read_json_body,
)

# Importing this module is intentionally fail-closed: DebugConfig.from_env()
# rejects missing/placeholder production tokens before a server can bind.
application = create_application()

__all__ = [
    "CSP",
    "STATIC_ALLOWLIST",
    "STATIC_MAP",
    "DebugApplication",
    "DebugWSGIApplication",
    "WSGIApplication",
    "application",
    "check_bearer_token",
    "create_application",
    "parse_content_length",
    "parse_json_body",
    "read_json_body",
]
