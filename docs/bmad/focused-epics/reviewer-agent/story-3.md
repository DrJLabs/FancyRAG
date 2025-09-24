# Story 1.3: Integrate Reviewer Into Focused-Epic Workflow

## Status

- Draft

## Story

**As the** BMAD workflow maintainer,
**I want** the reviewer persona wired into the focused-epic workflow with clear tracking,
**so that** teams run the reviewer automatically after development and capture findings in project docs.

## Acceptance Criteria

1. Update `.bmad-core/workflows/focused-epic.yaml` inserting a reviewer stage after the dev action, with guidance for skip conditions and inputs (`BMAD_REVIEWER_SKIP`, `BMAD_REVIEWER_STRICT`, `BMAD_REVIEWER_MODEL`).
2. Document reviewer usage in `docs/bmad/reviewer/README.md`, link from focused-epic guidance, and provide quick-start checklist + troubleshooting table.
3. Create or update `docs/bmad/issues/reviewer-rollout.md` to capture rollout status, owners, runtime/false-positive metrics, and telemetry links.
4. Publish telemetry metrics to the rollout tracker automatically after each focused-epic workflow run.
5. Invoke `tools/reviewer/preflight.sh` followed by `bmad reviewer run` within the workflow step, piping artifacts (`report.md`, `report.sarif`, `report.json`, `metrics.json`) for downstream QA tasks.
6. Introduce reviewer config toggle in `bmad-core/core-config.yaml` enabling skip for trivial diffs (with per-story overrides) and document rollback procedure.
7. Ensure reviewer outputs referenced by QA tasks (risk/test docs) via artifact hints and update templates accordingly.
8. Add optional dry-run matrix job (e.g., GitHub Actions) to surface reviewer summaries without gating merges.
9. Validate workflow changes via dry-run before merge to guarantee backwards compatibility and no secret requirements.
10. Provide rollback instructions and strict-mode governance (approval checklist, owners, telemetry thresholds).

## Tasks / Subtasks

- [ ] Modify `.bmad-core/workflows/focused-epic.yaml` to insert reviewer step after dev agent, referencing `tools/reviewer/preflight.sh` and `bmad reviewer run`, with skip/strict/model inputs documented. **Owner:** DevOps — **Due:** 2025-09-27 (AC 1,5)
- [ ] Add reviewer toggle block to `bmad-core/core-config.yaml` (e.g., `reviewer: { enabled: false, strict: false }`) with documentation in README. **Owner:** DevOps — **Due:** 2025-09-27 (AC 6,10)
- [ ] Update `docs/bmad/reviewer/README.md` with quick-start checklist, dependency install instructions, skip/strict guidance, telemetry troubleshooting, and rollback steps; cross-link from focused-epic docs. **Owner:** Analyst — **Due:** 2025-09-28 (AC 2)
- [ ] Extend focused-epic guidance (if applicable `docs/enhanced-ide-development-workflow.md`) with reviewer step instructions and skip criteria. (AC 2)
- [ ] Enhance `docs/bmad/issues/reviewer-rollout.md` to include telemetry auto-update section, strict-mode readiness checklist, and owner assignments. **Owner:** PM — **Due:** 2025-09-29 (AC 3,4,10)
- [ ] Implement telemetry sync script (`npm run reviewer:telemetry-sync`) that appends metrics after workflow runs. **Owner:** DevOps — **Due:** 2025-09-29 (AC 4)
- [ ] Update QA templates/gates to reference reviewer artifacts (risk/test design docs). (AC 7)
- [ ] Configure CI workflow (GitHub Actions) optional dry-run matrix executing reviewer step without gating merges, documenting how to enable/disable. **Owner:** DevOps — **Due:** 2025-09-29 (AC 8)
- [ ] Run workflow dry-run to validate compatibility; capture output and update change log. (AC 9)
- [ ] Document rollback procedure (config toggle off, workflow block revert) and strict-mode governance in README/tracker. (AC 6,10)

## Dev Notes

- **Workflow insertion:** Reviewer step should run after `dev: implement_story` and before QA gates, ensuring artifacts available for QA/test design references.
- **Skip strategy:** Default skip for doc-only diffs or changes <5 LOC; manual override via workflow input or config toggle.
- **Telemetry:** `npm run reviewer:telemetry-sync` reads latest `metrics.json` and appends aggregated metrics (runtime, high-severity count, false-positive rate) to rollout tracker table.
- **Documentation updates:** `docs/bmad/reviewer/README.md`, relevant sections in `docs/enhanced-ide-development-workflow.md`, and referencing links in `docs/user-guide.md`.
- **Governance:** Strict-mode readiness requires passing pilot metrics (runtime, false-positive thresholds) and approvals recorded in rollout tracker.
- **Rollback:** Document one-command toggle (config) and instructions to comment workflow step, plus note to retain documentation for future attempt.
- **Dependencies:** None beyond existing reviewer tooling; ensure instructions cover Semgrep/Jscpd installation for CI.

### Testing Standards

- Execute dry-run of focused-epic workflow verifying reviewer step executes, artifacts produced, skip flag honored, telemetry appended (`docs/bmad/focused-epics/reviewer-agent/story-3-test-design.md`).
- Validate documentation links (markdown-link-check), tracker tables, strict-mode checklist completion.

## Testing

- Dry-run workflow to confirm reviewer step ordering, skip flag, and artifact uploads.
- Skip scenario test ensuring `BMAD_REVIEWER_SKIP=1` bypasses reviewer step gracefully.
- Telemetry sync test verifying metrics appended to rollout tracker and strict-mode checklist updates.
- Documentation lint checks (markdown-link-check) ensuring references intact.

## Change Log

| Date       | Version | Description                                                                                                  | Author |
| ---------- | ------- | ------------------------------------------------------------------------------------------------------------ | ------ |
| 2025-09-24 | v0.1.0  | Story drafted aligning reviewer integration with focused-epic workflow, telemetry, and documentation updates | Codex  |

## Dev Agent Record

### Agent Model Used

- _To be completed by Dev agent._

### Debug Log References

- _To be completed by Dev agent._

### Completion Notes List

- _To be completed by Dev agent._

### File List

- _To be completed by Dev agent._

## QA Results

- _To be completed by QA agent after implementation._

## 🔬 Research & Validation Log (2025-09-24)

- **Researcher:** Dr. Evelyn Reed
- **Active Mode:** small-team
- **Primary Artifact:** docs/bmad/focused-epics/reviewer-agent/story-3.md
- **Summary:** Refined reviewer workflow integration story to detail config toggles, telemetry hooks, documentation updates, and rollback procedures required for team adoption.

### Findings & Actions

| Priority | Area                 | Recommended Change                                                            | Owner / Reviewer | Confidence | Mode       | Controls              | Evidence Location                      | Sources                                                                                                |
| -------- | -------------------- | ----------------------------------------------------------------------------- | ---------------- | ---------- | ---------- | --------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| High     | Workflow Integration | Add telemetry + dry-run matrix to focused-epic workflow before QA gate.       | DevOps / QA      | High       | small-team | ISO 25010 Reliability | .bmad-core/workflows/focused-epic.yaml | [GitHub Actions Reusable Workflows](https://docs.github.com/actions/using-workflows/reusing-workflows) |
| High     | Documentation        | Expand reviewer README with quick-start checklist + rollback steps.           | Analyst / PM     | High       | small-team | ASVS V14.2            | docs/bmad/reviewer/README.md           | [Semgrep OSS Deployment Guide](https://semgrep.dev/docs/getting-started/)                              |
| Medium   | Governance           | Link telemetry metrics + strict-mode readiness criteria into rollout tracker. | PM / QA          | Medium     | small-team | NIST SSDF PO.3        | docs/bmad/issues/reviewer-rollout.md   | [OpenTelemetry CI Patterns](https://opentelemetry.io/docs/ci/metrics/)                                 |

### Tooling Guidance

- **FOSS-first Recommendation:** Use GitHub Actions matrix to run reviewer in dry-run + strict modes without extra SaaS.
- **Paid Option (if required):** None; all steps rely on OSS tooling.
- **Automation / Scripts:** Introduce `npm run reviewer:telemetry-sync` to append metrics to rollout tracker post-run.

### Risk & Compliance Notes

- **Residual Risks:** CI environments lacking scanners (Medium) — mitigated via preflight install instructions.
- **Compliance / Control Mapping:** Aligns with ASVS secure automation and NIST SSDF deployment controls.
- **Monitoring / Observability:** Telemetry forwarded to rollout tracker and optional OpenTelemetry collector.
- **Rollback / Contingency:** Disable reviewer workflow step via config toggle; revert docs with instructions captured in README.

### Follow-Up Tasks

- [ ] Draft workflow snippet for reviewer dry-run matrix job — Owner: DevOps, Due: 2025-09-27
- [ ] Update docs/user-guide.md linking to reviewer quick-start — Owner: Analyst, Due: 2025-09-28

### Source Appendix

1. GitHub Actions Reusable Workflows — GitHub (Accessed 2025-09-24)
2. Semgrep OSS Deployment Guide — r2c (Accessed 2025-09-24)
3. OpenTelemetry CI Patterns — CNCF (Accessed 2025-09-24)

## 📝 Product Owner Validation (2025-09-24)

### Template Compliance Issues

- Story now follows BMAD template with status, tasks, Dev Notes, testing, change log, and agent placeholders.

### Critical Issues (Must Fix – Story Blocked)

- Pending: deliver workflow snippet, documentation updates, and telemetry automation before marking Ready for Dev.

### Should-Fix Issues

- Workflow snippet, documentation updates, telemetry sync, and rollout tracker enhancements are all scheduled with owners/dates in the task list; verify completion before marking Ready for Dev.
- Provide example dry-run matrix configuration once implemented.

### Nice-to-Have Improvements

- Consider publishing communications template (release note snippet) after initial rollout pilot.

### Anti-Hallucination Findings

- Requirements remain aligned with researcher/risk/test artifacts; no unsupported steps added.

### Final Assessment

- **Decision:** READY FOR DEV (after assigned workflow/documentation tasks complete)
- **Implementation Readiness Score:** 8/10
- **Confidence Level:** High — workflow integration plan now actionable and traceable
