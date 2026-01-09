"""Structured JSON logging helpers for the FancyRAG project."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


LOG_RECORD_SKIP_KEYS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


class JsonFormatter(logging.Formatter):
    """Simple JSON formatter ensuring structured log output."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key in LOG_RECORD_SKIP_KEYS or key.startswith("_"):
                continue
            payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        try:
            return json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError) as error:
            payload["serialization_error"] = str(error)
            return json.dumps(
                payload,
                ensure_ascii=False,
                default=lambda obj: f"<unserializable: {type(obj).__name__}>",
            )


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging with JSON output."""

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.propagate = False


__all__ = ["JsonFormatter", "configure_logging"]
