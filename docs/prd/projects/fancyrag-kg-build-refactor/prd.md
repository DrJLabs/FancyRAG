# Product Requirements Document: FancyRAG `kg_build.py` Monolith Decomposition and Repository Restructuring

## 1. Introduction

This document outlines the requirements for refactoring and reorganizing the FancyRAG codebase. The primary goal is to improve the maintainability, testability, and overall code quality of the repository by decomposing the monolithic `scripts/kg_build.py` script and restructuring the repository to separate library code from scripts.

## 2. Goals

- Improve the maintainability, testability, and overall code quality of the FancyRAG repository.
- Establish a clear and scalable project structure that facilitates future development and debugging.

## 3. Epics and User Stories

### Epic 1: `kg_build.py` Monolith Decomposition

- **User Story 1.1:** As a developer, I want to have a dedicated CLI module so that I can easily manage the command-line interface.
  - **Acceptance Criteria:**
    - A new file, `src/fancyrag/cli/kg_build_main.py`, is created.
    - This file contains all the argparse and wiring logic from `scripts/kg_build.py`.
    - The `if __name__ == "__main__":` block in `scripts/kg_build.py` calls the main function in this new module.
- **User Story 1.2:** As a developer, I want to have a dedicated pipeline module so that I can easily manage the GraphRAG pipeline orchestration.
  - **Acceptance Criteria:**
    - A new file, `src/fancyrag/kg/pipeline.py`, is created.
    - This file contains all the GraphRAG pipeline orchestration and dependency check logic from `scripts/kg_build.py`.
- **User Story 1.3:** As a developer, I want to have a dedicated splitter module so that I can easily manage the text splitter.
  - **Acceptance Criteria:**
    - A new file, `src/fancyrag/splitters/caching_fixed_size.py`, is created.
    - This file contains the `CachingFixedSizeSplitter` class from `scripts/kg_build.py`.
- **User Story 1.4:** As a developer, I want to have a dedicated QA evaluator module so that I can easily manage the QA evaluation logic.
  - **Acceptance Criteria:**
    - A new file, `src/fancyrag/qa/evaluator.py`, is created.
    - This file contains the `IngestionQaEvaluator`, thresholds, and totals math from `scripts/kg_build.py`.
- **User Story 1.5:** As a developer, I want to have a dedicated QA report module so that I can easily manage the QA report rendering.
  - **Acceptance Criteria:**
    - A new file, `src/fancyrag/qa/report.py`, is created.
    - This file contains the JSON/Markdown renderers from `scripts/kg_build.py`.
- **User Story 1.6:** As a developer, I want to have a dedicated Neo4j queries module so that I can easily manage the Neo4j queries.
  - **Acceptance Criteria:**
    - A new file, `src/fancyrag/db/neo4j_queries.py`, is created.
    - This file contains the query strings and minimal wrappers for counts and lookups from `scripts/kg_build.py`.
- **User Story 1.7:** As a developer, I want to have a dedicated schema module so that I can easily manage the schema loading and validation.
  - **Acceptance Criteria:**
    - A new file, `src/fancyrag/config/schema.py`, is created.
    - This file contains the default schema loader and validation logic from `scripts/kg_build.py`.

### Epic 2: Repository Restructuring

- **User Story 2.1:** As a developer, I want to move all core logic to a new `src/fancyrag/` package so that I can have a clear separation between library code and scripts.
  - **Acceptance Criteria:**
    - A new `src/fancyrag/` package is created.
    - All the new modules from Epic 1 are placed in this package.
- **User Story 2.2:** As a developer, I want `scripts/kg_build.py` to be a thin CLI entry point so that it is easy to understand and maintain.
  - **Acceptance Criteria:**
    - `scripts/kg_build.py` is modified to only contain the `if __name__ == "__main__":` block, which calls the main function in `src/fancyrag/cli/kg_build_main.py`.
- **User Story 2.3:** As a developer, I want to centralize all environment utilities so that they are easy to find and reuse.
  - **Acceptance Criteria:**
    - All environment utilities are moved to `src/fancyrag/utils/env.py`.
    - The existing tests for the environment utilities are updated to reflect the new location.

### Epic 3: Testing and Validation

- **User Story 3.1:** As a developer, I want to have unit tests for each new module so that I can ensure they are working correctly and prevent regressions.
  - **Acceptance Criteria:**
    - Unit tests are created for each of the new modules created in Epic 1.
    - The unit tests cover all the public functions in each module.
- **User Story 3.2:** As a developer, I want to have an end-to-end smoke test for the CLI so that I can quickly verify that the refactored application is working as expected.
  - **Acceptance Criteria:**
    - An end-to-end smoke test is created that invokes the `scripts/kg_build.py` CLI.
    - The smoke test verifies that the application runs without errors and produces the expected output.

## 4. Out of Scope

- New feature development.
- Major API changes.
- Infrastructure changes.

## 5. Assumptions

- The existing test suite is comprehensive and can be used to validate the refactored code.
- The current development team has the necessary skills and knowledge to complete the project.
