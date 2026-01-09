"""Tests for structured JSON logging configuration."""

from __future__ import annotations

import json
import logging
from io import StringIO

import pytest

from fancryrag.logging_setup import JsonFormatter, configure_logging

_TEST_EXCEPTION_MESSAGE = "Test exception"


class TestJsonFormatter:
    """Tests for the JsonFormatter class."""

    def test_format_basic_message(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_format_with_extra_fields(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.WARNING,
            pathname="test.py",
            lineno=42,
            msg="Warning message",
            args=(),
            exc_info=None,
        )
        record.user_id = "12345"
        record.request_id = "abc-def"

        output = formatter.format(record)
        data = json.loads(output)

        assert data["user_id"] == "12345"
        assert data["request_id"] == "abc-def"

    def test_format_skips_standard_fields(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Error message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert "pathname" not in data
        assert "lineno" not in data
        assert "funcName" not in data
        assert "process" not in data
        assert "thread" not in data

    def test_format_with_exception_info(self) -> None:
        formatter = JsonFormatter()
        def _raise_test_exception() -> None:
            raise ValueError(_TEST_EXCEPTION_MESSAGE)
        try:
            _raise_test_exception()
        except ValueError:
            import sys
            exc_info = sys.exc_info()
            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=42,
                msg="Error with exception",
                args=(),
                exc_info=exc_info,
            )

            output = formatter.format(record)
            data = json.loads(output)

            assert "exc_info" in data
            assert "ValueError: Test exception" in data["exc_info"]

    def test_format_skips_private_fields(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Message",
            args=(),
            exc_info=None,
        )
        record._private_field = "should not appear"

        output = formatter.format(record)
        data = json.loads(output)

        assert "_private_field" not in data

    def test_format_handles_non_serializable_fields(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Message",
            args=(),
            exc_info=None,
        )
        record.non_serializable = object()

        output = formatter.format(record)
        data = json.loads(output)

        assert "serialization_error" in data
        assert data["message"] == "Message"

    def test_format_timestamp_is_utc_iso_format(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert "timestamp" in data
        assert "T" in data["timestamp"]
        assert data["timestamp"].endswith("+00:00") or data["timestamp"].endswith("Z")

    def test_format_preserves_unicode(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Unicode: 日本語 🚀",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["message"] == "Unicode: 日本語 🚀"


class TestConfigureLogging:
    """Tests for the configure_logging function."""

    def test_configure_logging_sets_json_formatter(self) -> None:
        configure_logging()

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 1
        handler = root_logger.handlers[0]
        assert isinstance(handler.formatter, JsonFormatter)

    def test_configure_logging_sets_level(self) -> None:
        configure_logging(level=logging.DEBUG)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_configure_logging_default_level_is_info(self) -> None:
        configure_logging()

        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_configure_logging_outputs_to_stdout(self) -> None:
        import sys

        configure_logging()

        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]
        assert handler.stream == sys.stdout

    def test_configure_logging_replaces_existing_handlers(self) -> None:
        root_logger = logging.getLogger()
        initial_handler = logging.StreamHandler()
        root_logger.addHandler(initial_handler)

        configure_logging()

        assert len(root_logger.handlers) == 1
        assert root_logger.handlers[0] is not initial_handler

    def test_configure_logging_disables_propagation(self) -> None:
        configure_logging()

        root_logger = logging.getLogger()
        assert root_logger.propagate is False

    def test_configured_logger_outputs_valid_json(self, capsys) -> None:
        configure_logging()
        logger = logging.getLogger("test.module")

        logger.info("test message", extra={"key": "value"})

        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())

        assert data["message"] == "test message"
        assert data["key"] == "value"
        assert data["level"] == "INFO"

    def test_configured_logger_multiple_messages(self, capsys) -> None:
        configure_logging()
        logger = logging.getLogger("test")

        logger.info("first")
        logger.warning("second")
        logger.error("third")

        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")

        assert len(lines) == 3
        first = json.loads(lines[0])
        second = json.loads(lines[1])
        third = json.loads(lines[2])

        assert first["message"] == "first"
        assert first["level"] == "INFO"
        assert second["message"] == "second"
        assert second["level"] == "WARNING"
        assert third["message"] == "third"
        assert third["level"] == "ERROR"

    def test_configured_logger_debug_level(self, capsys) -> None:
        configure_logging(level=logging.DEBUG)
        logger = logging.getLogger("test")

        logger.debug("debug message")
        logger.info("info message")

        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")

        assert len(lines) == 2

    def test_configured_logger_warning_level_filters_info(self, capsys) -> None:
        configure_logging(level=logging.WARNING)
        logger = logging.getLogger("test")

        logger.info("info message")
        logger.warning("warning message")

        captured = capsys.readouterr()
        lines = [line for line in captured.out.strip().split("\n") if line]

        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["level"] == "WARNING"