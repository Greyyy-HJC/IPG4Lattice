# AGENTS.md

Project-specific instructions for coding agents working in this repository.

## Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

- State assumptions explicitly.
- If multiple interpretations exist, present them instead of choosing silently.
- If requirements are unclear, ask before implementing.
- If a simpler approach exists, call it out.

## Simplicity First

Write the minimum code that solves the requested problem.

- No features beyond what was asked.
- No speculative abstractions.
- No unnecessary configurability.
- Prefer locally understandable logic over framework-heavy design.

## Surgical Changes

Touch only what is required for the task.

- Do not refactor unrelated code unless asked.
- Match existing repository style and conventions.
- Avoid broad formatting-only changes.
- Clean up only unused code introduced by your own edits.

## Goal-Driven Execution

Turn tasks into verifiable outcomes.

- Define expected outputs before editing.
- Run focused checks or scripts relevant to the change.
- For multi-step work, keep a short plan and verify each step.
- Do not stop at code edits; validate behavior when possible.

## Workflow Hygiene

- Before each `git add`/`git commit`, verify `.gitignore` still covers local artifacts.
- After each meaningful implementation pass, check whether `PROJECT_LOG.md` needs an appended entry.
- Keep generated local data (`ensemble/`, `artifacts/`) out of git unless explicitly requested.

## Project-Specific Rules

- Use repository-root `.venv` as the default Python environment (`python3 -m venv .venv`).
- Keep `requirements.txt` aligned with direct Python dependencies needed by scripts and tests.
- Treat `environment.yml` as an optional machine-specific template for CUDA/MPI stacks, not the default workflow.
- Preserve input CG ensembles; write IPG outputs to separate output roots.
- Keep validation and analysis scripts reproducible with explicit CLI options.

## Documentation Maintenance

- Keep `SPEC.md` aligned with structural or workflow changes.
- Append meaningful progress and decisions to `PROJECT_LOG.md`.
- Keep `README.md` accurate for setup, usage, and repository layout.
