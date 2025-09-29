# Epic 1: Environment & Workspace — Local Stack Bootstrap

## Epic Goal
Provision a reproducible Python workspace for GraphRAG that operators can bootstrap quickly, ensuring required packages install cleanly and environment variables default to the project-owned Neo4j and Qdrant containers.

## Epic Description
**Existing System Context:**
- Current relevant functionality: No dedicated environment yet; operators previously targeted managed services manually.
- Technology stack: Python 3.12, `neo4j-graphrag[experimental,openai,qdrant]`, Docker Compose neo4j/qdrant stack, OpenAI GPT models.
- Integration points: CLI tooling must authenticate to Neo4j, Qdrant, and OpenAI using environment variables sourced from `.env`.

**Enhancement Details:**
- What's being added/changed: Standardize virtual environment bootstrap scripts, dependency management, and environment variable templates aligned with the new Docker Compose stack (`docker-compose.neo4j-qdrant.yml`).
- How it integrates: Creates repeatable install scripts and configuration that point to local services by default while allowing overrides for managed deployments without code changes.
- Success criteria: Operators can create the venv, install `neo4j-graphrag[experimental,openai,qdrant]`, validate imports, populate `.env` with local defaults, and run diagnostics within 30 minutes.

## Stories
1. **Story 1:** Author workspace bootstrap script that installs Python 3.12 dependencies, pins GraphRAG extras, and scaffolds `.env` with local service defaults.
2. **Story 2:** Provide `.env` template and documentation covering OpenAI, Neo4j, and Qdrant configuration with guardrails for local vs. managed endpoints.
3. **Story 3:** Create verification command/tests that confirm module imports, Compose service reachability, and record package versions for auditing.

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

- The bootstrap workflow must install Python dependencies, activate the virtualenv, and populate `.env` defaults pointing at `neo4j://localhost:7687` and `http://localhost:6333` sourced from the compose stack.
- Integration points: `.env` variables for OpenAI API access, Neo4j Bolt credentials, and Qdrant URL/API key. Document how to swap to managed endpoints without editing code.
- Existing patterns to follow: CLI-only workflows, structured logging, pinned dependencies, diagnostics under `cli.diagnostics`.
- Critical compatibility requirements: Do not hard-code secrets; ensure validation scripts succeed with the Dockerized services before story completion.

The epic should maintain system integrity while delivering a reliable, bootstrapped workspace for operators preparing to ingest and retrieve with GraphRAG."

### Story Override Workflow
- Default rule: do not create the next story while the previous one is not marked `Done`.
- When an override is unavoidable (e.g., downstream scheduling pressure), run `python -m cli.stories --override-incomplete --stories-dir docs/stories --new-story <new_story_path> --reason "<why>"` to:
  - Log actor, timestamp, prior story id, and reason into `docs/bmad/story-overrides.md`.
  - Inject a risk acknowledgement note under `## Dev Notes` in the new story document.
- Overrides require Product Owner awareness and QA follow-up; ensure the `reason` captures mitigating actions and link back to the prior story in status meetings.
