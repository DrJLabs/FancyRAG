# Project Agents

This file provides guidance and memory for Codex CLI.

## Workflow Rules

- Use normal git branches for work; do not use git worktrees for isolation.
- After any update/push to an existing PR branch, post review triggers as separate comments: `@codex review` and `/gemini review` (initial PR creation does not require this).

## Testing Expectations

- Always run the local-stack smoke mirror after changes that touch env handling, local-stack scripts, compose files, or the `local-stack-smoke` workflow: `scripts/run_local_stack_smoke.sh` (reads `OPENAI_API_KEY` from the environment or `.env.local`/`.env`).
- At minimum, run `scripts/check_local_stack.sh --config` before pushing changes that affect Docker compose or smoke automation.
