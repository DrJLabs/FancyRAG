"""Namespace compatibility checks for the fancyrag transition."""

from __future__ import annotations

import importlib
import sys


def test_fancryrag_aliases_fancyrag() -> None:
    fancyrag = importlib.import_module("fancyrag")
    fancryrag = importlib.import_module("fancryrag")
    assert sys.modules["fancryrag"] is sys.modules["fancyrag"]
    assert fancryrag is fancyrag


def test_fancryrag_submodules_alias_fancyrag() -> None:
    runtime = importlib.import_module("fancryrag.mcp.runtime")
    canonical = importlib.import_module("fancyrag.mcp.runtime")
    assert runtime is canonical
