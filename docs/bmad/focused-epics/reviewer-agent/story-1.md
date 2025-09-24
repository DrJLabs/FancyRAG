# Story 1.1: Establish Free Scanner Baseline

## Status

- Done

## Story

**As a** BMAD developer,
**I want** built-in free scanners that capture security smells and duplication,
**so that** the reviewer persona has concrete evidence before critiquing diffs.

## Acceptance Criteria

1. Provide a bash entrypoint `tools/reviewer/preflight.sh` that checks for Semgrep/Jscpd binaries, pins minimum CLI versions (Semgrep ≥1.86.0, Jscpd ≥3.5.4), and prepares cache directories.
2. Add a Node.js script `tools/reviewer/collect-scans.js` that runs Semgrep (with diff-aware baseline commit) and Jscpd with a 60% threshold, emitting JSON and SARIF summaries.
3. Generate churn metadata for touched files from the last 30 days (`git log`) saved beside scan results and labeled with repository slug + timestamp for correlation.
4. Persist tooling metadata (CLI versions, execution duration, diff coverage, cache size) to `artifacts/reviewer/<timestamp>/metrics.json` for downstream reviewer weighting and telemetry.
5. Reviewer scripts run using existing Node.js runtime without additional global installs; allow overrides via `SEMgrep_BIN` / `JSCPD_BIN` environment variables for air-gapped runners.
6. All artifacts write to `.bmad-cache/reviewer/{git_sha}/` to prevent collisions and mirror to `artifacts/reviewer/<timestamp>/` when executed.
7. Workflow exit codes remain zero when scans succeed; failures surface actionable remediation tips.
8. Semgrep and Jscpd configs live under version control (e.g., `tools/reviewer/semgrep.yaml`, `tools/reviewer/jscpd.json`) with documented ignore patterns and diff-only defaults.
9. Scripts include unit-friendly structure (pure functions exported for future tests) and capture structured logs (JSON Lines) for QA replay.
10. README updates document prerequisites, troubleshooting, cache pruning, and guidance for calibrating ignore rules based on runtime metrics.

## Tasks / Subtasks

- [x] Author `tools/reviewer/preflight.sh` with binary checks, minimum version assertions, trusted-install warnings, cache directory bootstrap, and override support for `SEMgrep_BIN` / `JSCPD_BIN`. (AC 1,5)
- [x] Implement `tools/reviewer/collect-scans.js` to orchestrate Semgrep (diff-aware baseline commit) and Jscpd (60% threshold), enforce 180s/120s timeouts, capture stdout/stderr, and serialize JSON/SARIF outputs into `.bmad-cache/reviewer/{git_sha}/`. (AC 2,6,7,8,9)
- [x] Generate churn metadata (30-day window) with repository slug + timestamp, persist as `churn.json`, and append diff coverage and runtime metrics to `metrics.json`. (AC 3,4,6)
- [x] Add cache-pruning utility (e.g., npm script or docs command) to keep `.bmad-cache/reviewer/` under 250 MB and document the procedure. **Owner:** Dev — **Due:** 2025-09-26 (AC 4,6,10)
- [x] Update `docs/bmad/reviewer/README.md` with prerequisites, cache management, strict-mode guidance, and troubleshooting, including references to new scripts. **Owner:** Dev — **Due:** 2025-09-26 (AC 8,9,10)
- [x] Wire optional npm script `npm run reviewer:scan` and document usage in README, ensuring exit codes stay zero on success and propagate actionable errors on failure. **Owner:** QA — **Due:** 2025-09-27 (AC 7,10)

## Dev Notes

- **Integration points:**
  - Scripts reside in `tools/reviewer/` alongside config files (`semgrep.yaml`, `jscpd.json`).
  - Cache root directories: `.bmad-cache/reviewer/{git_sha}/` for working data, `artifacts/reviewer/<timestamp>/` for published copies.
  - Metrics file must include CLI versions, execution duration, diff coverage %, cache footprint, and finding counts.
- **Configuration:**
  - Default Semgrep command `semgrep scan --config auto --baseline --sarif --json --timeout 180` with repo override path appended when present.
  - Jscpd runs via `npx jscpd@^3.5.4 --threshold 60 --gitignore --format json --output <cache>/jscpd.json`.
  - Allow overrides via `SEMgrep_BIN` and `JSCPD_BIN` for air-gapped runners; log override usage in preflight output.
- **Error handling & observability:**
  - Preflight should exit non-zero with descriptive messages when binaries missing/outdated.
  - Collect script logs JSON Lines (`log.jsonl`) capturing command, args, duration, stdout length, and exit code for QA replay.
  - Metrics must warn when runtime >3 minutes or cache size >250 MB (aligns with risk mitigations).
- **Dependencies:** Node.js ≥18 (already project requirement), Semgrep OSS CLI ≥1.86.0, `npx jscpd@^3.5.4`.
- **Rollbacks:** To bypass reviewer scans set `BMAD_REVIEWER_SKIP=1`; document fallback within README.

### Testing Standards

- Follow guidance in `docs/bmad/focused-epics/reviewer-agent/story-1-test-design.md`.
- Implement unit tests for preflight (via `bats`) and integration tests validating diff-mode enforcement, override paths, cache pruning, and end-to-end execution.

## Testing

- Integration suite executing `tools/reviewer/collect-scans.js` on fixture repo verifying JSON/SARIF outputs, metrics content, and runtime thresholds.
- Failure simulation ensuring Semgrep errors bubble up with actionable messaging.
- Cache pruning tests generating aged entries (>7 days) confirming cleanup.
- End-to-end npm alias run verifying artifacts sync between cache and `artifacts/reviewer/<timestamp>/`.

## Change Log

| Date       | Version | Description                                                                            | Author    |
| ---------- | ------- | -------------------------------------------------------------------------------------- | --------- |
| 2025-09-24 | v0.1.0  | Initial focused-epic story draft with scanner baseline scope                           | Codex     |
| 2025-09-24 | v1.0.0  | Implemented scanner tooling, metrics, cache pruning utility, and documentation updates | Dev Agent |

## Dev Agent Record

### Agent Model Used

- Codex Dev Agent (GPT-5 based)

### Debug Log References

- `artifacts/reviewer/20250924T192915Z/log.jsonl`
- `artifacts/reviewer/20250924T192915Z/metrics.json`

### Completion Notes List

- Delivered Semgrep/Jscpd orchestrator with diff-aware baseline, churn, and telemetry alerts.
- Added cache pruning and metrics validation commands with npm aliases.
- Ran reviewer preflight + scan, capturing artifacts under `artifacts/reviewer/20250924T192915Z/`.

### File List

- tools/reviewer/preflight.sh
- tools/reviewer/collect-scans.js
- tools/reviewer/prune-cache.js
- tools/reviewer/semgrep.yaml
- tools/reviewer/jscpd.json
- docs/bmad/reviewer/README.md
- package.json

## QA Results

- 2025-09-24: QA executed `npm run reviewer:validate` against `artifacts/reviewer/20250924T193704Z/metrics.json`. Validation PASS; runtime 26.22s, cache 4.48MB. QA Gate PASS — `docs/bmad/qa/assessments/story-1.1-qa-gate-20250924.md` confirms CI + prune automation.

## 🔬 Research & Validation Log (2025-09-24)

- **Researcher:** Dr. Evelyn Reed
- **Active Mode:** small-team
- **Primary Artifact:** docs/bmad/focused-epics/reviewer-agent/story-1.md
- **Summary:** Validated free scanner baseline plan against current Semgrep/Jscpd guidance, emphasizing diff-only scans, cache scoping, and runtime telemetry to keep reviewer findings actionable for shared teams.

### Findings & Actions

| Priority | Area                   | Recommended Change                                                                                                                                      | Owner / Reviewer | Confidence | Mode       | Controls                  | Evidence Location                    | Sources                                                                                                                             |
| -------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | ---------- | ---------- | ------------------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| High     | Security Coverage      | Require Semgrep `--baseline` diff mode with CLI version floor 1.86.0 and capture runtime metrics for visibility.                                        | Dev / QA         | High       | small-team | OWASP ASVS V14.2          | artifacts/reviewer/<ts>/metrics.json | [Semgrep CLI Reference](https://semgrep.dev/docs/cli-reference/scan/)                                                               |
| Medium   | Maintainability        | Scope cache paths by repository SHA and prune artifacts older than seven days to prevent collisions in shared runners.                                  | Dev / SM         | Medium     | small-team | ISO 25010 Maintainability | `.bmad-cache/reviewer/`              | [GitHub Actions Caching Best Practices](https://docs.github.com/actions/using-workflows/caching-dependencies-to-speed-up-workflows) |
| Medium   | Operational Resilience | Emit structured JSON logs and allow env overrides for Semgrep/Jscpd binaries to support air-gapped runners.                                             | DevOps / QA      | Medium     | small-team | NIST SP 800-218           | tools/reviewer/preflight.sh          | [Jscpd Advanced Usage](https://jscpd.dev/docs/advanced-usage)                                                                       |
| High     | CI Integration         | Add GitHub Actions step installing Semgrep CLI via pip and enforce reviewer scan + metrics validation on PRs to meet modern diff-aware gating guidance. | DevOps           | High       | small-team | OWASP ASVS V14.4          | .github/workflows/pr-validation.yaml | [Semgrep Sample CI Config](https://semgrep.dev/docs/semgrep-ci/sample-ci-configs)                                                   |
| Medium   | QA Automation          | Generate synthetic cache fixtures (>7 days, >250 MB) and automate prune validation to keep reviewer telemetry deterministic.                            | QA               | Medium     | small-team | ISO 25010 Reliability     | tools/reviewer/prune-cache.test.js   | Internal Fixture Plan                                                                                                               |

### Tooling Guidance

- **FOSS-first Recommendation:** Use Semgrep OSS CLI and `npx jscpd@latest` with pinned versions recorded in `metrics.json`.
- **Paid Option (if required):** Semgrep Pro — only when enterprise policies mandate managed rule hosting; document license impact.
- **Automation / Scripts:** Add `npm run reviewer:scan -- --metrics` wrapper to streamline diff-mode execution and metrics capture.

### Risk & Compliance Notes

- **Residual Risks:** Runtime spikes on repos >150k LoC (Medium) until ignore lists tuned; cache bloat (Low) mitigated by seven-day pruning.
- **Compliance / Control Mapping:** Aligns with ASVS V14 secure SDLC controls and NIST SSDF practices for automated analysis.
- **Monitoring / Observability:** Record runtime + findings counts via `metrics.json`; emit logs to CI artifacts for dashboards.
- **Rollback / Contingency:** Disable Semgrep/Jscpd invocation via `BMAD_REVIEWER_SKIP=1` and remove metrics generation if failures persist.

### Follow-Up Tasks

- [x] Add metrics validation guard via `npm run reviewer:validate` and record pass criteria — Owner: QA, Completed: 2025-09-24

### Source Appendix

1. Semgrep CLI Reference — r2c (Accessed 2025-09-24)
2. GitHub Actions Caching Best Practices — GitHub (Accessed 2025-09-24)
3. Jscpd Advanced Usage — OSS Maintainers (Accessed 2025-09-24)

## 📝 Product Owner Validation (2025-09-24)

### Template Compliance Issues

- Story continues to follow the BMAD story template with downstream agent sections complete.
- Reviewer automation cross-referenced in `.github/workflows/pr-validation.yaml` and `docs/bmad/reviewer/README.md` per workflow policy.

### Critical Issues (Must Fix – Story Blocked)

- None (validated 2025-09-24 after CI + prune automation landed).

### Should-Fix Issues

- Monitor reviewer runtime telemetry post-merge; adjust ignore lists if CI runtime trends above 180 seconds.

### Nice-to-Have Improvements

- Capture sample ignore configs during Story 1.2 rollout to accelerate tuning for large repos.

### Anti-Hallucination Findings

- All requirements trace back to researcher/test designs and public tooling docs; no unsupported claims observed.

### Final Assessment

- **Decision:** READY FOR PR (GO)
- **Implementation Readiness Score:** 10/10
- **Confidence Level:** High — Story 1.1 tooling, docs, CI gate, and QA evidence are complete
