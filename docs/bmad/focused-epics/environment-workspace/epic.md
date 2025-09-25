# Epic 1: Environment & Workspace — Brownfield Enhancement

## Epic Goal
Provision a reproducible Python workspace for GraphRAG that operators can bootstrap quickly, ensuring required packages install cleanly and environment variables are documented before deeper feature work begins.

## Epic Description
**Existing System Context:**
- Current relevant functionality: No dedicated environment yet; operators run ad-hoc scripts against managed Neo4j and Qdrant services.
- Technology stack: Python 3.12, Neo4j 5.x, Qdrant 1.10+, OpenAI GPT models.
- Integration points: Local CLI tooling must authenticate to Neo4j, Qdrant, and OpenAI using environment variables.

**Enhancement Details:**
- What's being added/changed: Standardize virtual environment bootstrap scripts, dependency management, and environment variable templates for GraphRAG workflows.
- How it integrates: Creates repeatable install scripts and configuration that align with the existing managed services without altering their endpoints.
- Success criteria: Operators can create the venv, install `neo4j-graphrag[openai,qdrant]`, validate imports, and populate a documented `.env` template with required secrets within 30 minutes.

## Stories
1. **Story 1:** Author workspace bootstrap script that installs Python 3.12 dependencies and pins GraphRAG extras.
2. **Story 2:** Provide `.env` template and documentation covering OpenAI, Neo4j, and Qdrant configuration with guardrails.
3. **Story 3:** Create verification command/tests that confirm module imports and record package versions for auditing.

## Compatibility Requirements
- [x] Existing APIs remain unchanged
- [x] Database schema changes are backward compatible
- [x] UI changes follow existing patterns
- [x] Performance impact is minimal

## Risk Mitigation
- **Primary Risk:** Mismatched dependency versions leading to import/runtime failures.
- **Mitigation:** Pin required package versions in lockfile and provide automated sanity check script.
- **Rollback Plan:** Revert to previous dependency versions and uninstall new scripts if validation fails.

## Definition of Done
- [x] All stories completed with acceptance criteria met
- [x] Existing functionality verified through testing
- [x] Integration points working correctly
- [x] Documentation updated appropriately
- [x] No regression in existing features

## Validation Checklist
**Scope Validation:**
- [x] Epic can be completed in 1-3 stories maximum
- [x] No architectural documentation is required
- [x] Enhancement follows existing patterns
- [x] Integration complexity is manageable

**Risk Assessment:**
- [x] Risk to existing system is low
- [x] Rollback plan is feasible
- [x] Testing approach covers existing functionality
- [x] Team has sufficient knowledge of integration points

**Completeness Check:**
- [x] Epic goal is clear and achievable
- [x] Stories are properly scoped
- [x] Success criteria are measurable
- [x] Dependencies are identified

## Story Manager Handoff
"Please develop detailed user stories for this brownfield epic. Key considerations:

- This enhancement standardizes the Python CLI environment powering our GraphRAG workflows built on Python 3.12 with `neo4j-graphrag`, Neo4j 5.x, Qdrant 1.10+, and OpenAI GPT models.
- Integration points: `.env` variables for OpenAI API access, Neo4j Bolt credentials, and Qdrant API/URL configuration.
- Existing patterns to follow: CLI-only workflows, structured logging, pinned dependencies.
- Critical compatibility requirements: Do not modify managed service endpoints; maintain reproducible installation and ensure validation scripts succeed without exposing secrets.

The epic should maintain system integrity while delivering a reliable, bootstrapped workspace for operators preparing to ingest and retrieve with GraphRAG."

### Story Override Workflow
- Default rule: do not create the next story while the previous one is not marked `Done`.
- When an override is unavoidable (e.g., downstream scheduling pressure), run `python -m cli.stories --override-incomplete --stories-dir docs/stories --new-story <new_story_path> --reason "<why>"` to:
  - Log actor, timestamp, prior story id, and reason into `docs/bmad/story-overrides.md`.
  - Inject a risk acknowledgement note under `## Dev Notes` in the new story document.
- Overrides require Product Owner awareness and QA follow-up; ensure the `reason` captures mitigating actions and link back to the prior story in status meetings.
