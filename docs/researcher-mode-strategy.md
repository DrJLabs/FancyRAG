# Researcher Agent Refactor Plan

## Objectives

- Introduce selectable operating modes tuned for solo developers (default), small teams, and enterprise programs.
- Strengthen sourcing, compliance, and documentation practices across all research work.
- Keep workflows efficient while favoring free and open-source tooling where practical.

## Key Improvements

- **Mode awareness:** Add mode negotiation on activation with default `solo`; allow user to switch to `small-team` or `enterprise` explicitly.
- **Hygiene baseline:** Require standardized research log entries, de-duplication of findings, and explicit confidence + risk notes.
- **Source governance:** Mandate cross-checking of claims with at least two credible sources when risk is medium/high.
- **Security posture:** Elevate guidance for threat modeling, dependency risk, and data handling in higher modes.
- **Collaboration assets:** Share reusable prompt, log, and checklist assets to align downstream agents.
- **FOSS-first tooling:** Provide vetted open-source options before suggesting paid services; flag when paid vendors are unavoidable.

## Mode Profiles

| Mode         | Default Audience                    | Scope & Depth                                                                            | Collaboration                                                 | Risk & Compliance                                                | Output Targets                                              |
| ------------ | ----------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------- |
| `solo`       | Solo devs and indie maintainers     | Focus on actionable, low-overhead upgrades; highlight quick wins and critical fixes only | Lightweight notes; optional handoff packets                   | Emphasize security/privacy basics                                | Concise research log with prioritized next steps            |
| `small-team` | Squads ≤10 engineers                | Balance depth with maintainable processes; ensure shared understanding                   | Shared research workspace outline; highlight owner + reviewer | Introduce dependency tracking and lightweight policy checks      | Research log + change summary ready for team review         |
| `enterprise` | Regulated / security-sensitive orgs | Exhaustive validation; ensure compliance and auditability                                | Formal documentation, handoff checklists, RACI callouts       | Require compliance references, threat models, and rollback plans | Mode-specific dossier including risk register and approvals |

## Artifacts to Update

1. `.bmad-core/agents/researcher.md`
2. `.bmad-core/tasks/validate-plan-with-research.md`
3. `.bmad-core/tasks/create-deep-research-prompt.md`
4. `bmad-core/checklists/researcher-checklist.md` (new)
5. Supporting docs/templates as needed (e.g., research log scaffolding)

## Rollout & Success Metrics

- Feature gating: ship the mode-aware flow behind a config flag with `solo` as the default fallback.
- Deployment plan: dry run with internal champions → small cohort rollout → org-wide enablement.
- KPIs: reduced time-to-decision, lower defect rate in validated artifacts, higher citation completeness, fewer audit findings post-merge.
- Rollback: toggle the config flag off and restore prior templates if regressions surface.

## Open Questions

- Should workflows auto-switch mode based on artifact tags? → Future enhancement.
- Do we need a distribution copy under `dist/agents`? → Evaluate during implementation.
