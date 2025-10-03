# Project Brief: FancyRAG `kg_build.py` Monolith Decomposition and Repository Restructuring

## 1. Project Title
FancyRAG `kg_build.py` Monolith Decomposition and Repository Restructuring

## 2. Project Goals

- **Primary Goal:** Improve the maintainability, testability, and overall code quality of the FancyRAG repository.
- **Secondary Goal:** Establish a clear and scalable project structure that facilitates future development and debugging.

## 3. Project Scope

### In Scope

- **Decomposition of `scripts/kg_build.py`:** This monolithic script will be broken down into smaller, single-responsibility modules. The new modules will be organized under a new `src/fancyrag/` package, with `scripts/kg_build.py` becoming a thin command-line interface (CLI) entry point. The proposed module split is as follows:
  - `src/fancyrag/cli/kg_build_main.py`: CLI argument parsing and wiring.
  - `src/fancyrag/kg/pipeline.py`: GraphRAG pipeline orchestration and dependency checks.
  - `src/fancyrag/splitters/caching_fixed_size.py`: `CachingFixedSizeSplitter` class.
  - `src/fancyrag/qa/evaluator.py`: `IngestionQaEvaluator`, thresholds, and totals math.
  - `src/fancyrag/qa/report.py`: JSON/Markdown renderers.
  - `src/fancyrag/db/neo4j_queries.py`: Neo4j query strings and minimal wrappers.
  - `src/fancyrag/config/schema.py`: Default schema loader and validation.
- **Repository Reorganization:**
  - All core logic will be moved from the `scripts/` directory to the `src/fancyrag/` package.
  - The `scripts/` directory will only contain thin entry-point scripts that call the logic in `src/fancyrag/`.
- **Establish Guardrails:**
  - Enforce a clear separation between "library code" (under `src/`) and "scripts."
  - Centralize environment utilities in `fancyrag.utils.env`.
  - Aim for modules between 200-400 lines of code (LOC), each with a single reason to change.
  - Public functions should have five or fewer parameters and use typed dataclasses for return values where appropriate.
- **Testing:**
  - Unit tests will be created for each new module.
  - One end-to-end smoke test will be created to invoke the CLI and ensure the refactored application works as expected.

### Out of Scope

- **New Feature Development:** This project is focused solely on refactoring and reorganization. No new features will be added.
- **Major API Changes:** The public-facing API of the service will remain unchanged.
- **Infrastructure Changes:** No changes will be made to the existing infrastructure.

## 4. Success Metrics

### Code Quality

- Reduced code complexity in `scripts/kg_build.py`.
- Improved code cohesion and reduced coupling in the new modules.
- Adherence to the new coding standards and guardrails.

### Testability

- Increased unit test coverage for the refactored code.
- The ability to test individual components in isolation.

### Maintainability

- Easier to understand and debug the codebase.
- Reduced "blast radius" for future changes.

## 5. Assumptions and Constraints

### Assumptions

- The existing test suite is comprehensive and can be used to validate the refactored code.
- The current development team has the necessary skills and knowledge to complete the project.

### Constraints

- The project must be completed without disrupting the existing functionality of the application.
- The project must be completed within the agreed-upon timeline and budget.
