# Fancryrag Coding Standards

These guidelines keep the codebase consistent for AI and human contributors. Follow them for all Python, configuration, and infrastructure changes unless a story or architecture decision mandates an exception.

## General Principles
- Prefer clarity over cleverness. Optimize for maintainability and observability first.
- Treat lint errors and warnings as failures; fix or justify in the PR.
- All changes should include corresponding documentation updates when behavior or operations change.

## Python Style
- **Language & Runtime**: Python 3.12 (managed by Astral `uv`).
- **Formatting**: Use `ruff format` (PEP 8 compatible with 4-space indentation, 88-character soft limit). Do not hand-edit formatting that the formatter would change.
- **Linting**: Run `ruff check` before pushing; address or explicitly ignore with inline comments referencing justification.
- **Imports**: Use absolute imports within the `fancryrag` package. Group in order: standard library, third-party, application, each separated by a blank line.
- **Type Hints**: Required on all public functions, class methods, and module-level variables. Use `typing` features (e.g., `TypedDict`, `Protocol`) for complex structures.
- **Naming Conventions**:
  - Modules, packages, and files: `snake_case`.
  - Variables & functions: `lower_snake_case`.
  - Classes & Exceptions: `CapWords`.
  - Constants: `UPPER_SNAKE_CASE`.
  - Private members: single leading underscore.
- **Docstrings**: Apply Google-style docstrings on public functions, classes, and modules when logic is non-trivial or exposed outside the module.
- **Logging**: Use the standard `logging` library. Never print directly except in CLI utilities where stdout is the interface.
- **Error Handling**: Raise specific exceptions; include actionable messages. Wrap external service calls (Neo4j, embeddings) with retries and meaningful context.
- **Asynchronous Code**: Prefer `async`/`await` for I/O-bound tasks. Document event loop expectations in docstrings.

## Configuration & Environment
- Keep secrets out of source control. Reference environment variables (`.env.local`) via `python-dotenv`.
- Validate critical environment variables at startup; fail fast with descriptive errors.
- Document new variables in `.env.example` and relevant operations guides.

## Testing
- Use `pytest` for all automated tests.
- Organize tests mirroring source modules under `tests/` (e.g., `tests/tools/test_run_pipeline.py`).
- Provide fixtures for Neo4j or external API interactions using Testcontainers or mocks.
- Every bug fix requires a regression test.

## Documentation & Comments
- Keep comments high-value and up-to-date; remove commented-out code.
- Update `README.md`, `docs/architecture.md`, or operations docs when workflows change.
- Reference architecture decisions (ADR/PR link) in comments when deviating from defaults.

## Git & Commit Hygiene
- Follow Conventional Commit messages (`feat:`, `fix:`, `chore:`, etc.).
- Commit only formatted, linted code; avoid committing `.env.local` or generated artifacts.
- Review diffs to ensure no secrets or tokens slip into history.

## MCP & Neo4j Specific Guidelines
- Encapsulate Cypher queries in dedicated modules/functions with parameterized inputs; avoid string interpolation for user-supplied values.
- Use GraphRAG helper utilities for index creation and retrieval primitives to reduce drift against upstream library updates.
- Ensure FastMCP tool definitions return JSON-serializable structures with consistent keys and types.
