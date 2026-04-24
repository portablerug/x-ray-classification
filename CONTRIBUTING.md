# Contributing Guide

## Branch Naming

Use descriptive branch names:

- `feature/<short-description>`
- `fix/<short-description>`
- `docs/<short-description>`
- `chore/<short-description>`

Examples:
- `feature/phase1-bigquery-metadata`
- `docs/update-roadmap-phase2`

## Commit Message Style

Use concise, action-based messages:

- `add phase 1 cloud setup checklist`
- `update label mapping assumptions`
- `fix metric naming in evaluation notes`

## Pull Request Requirements

Each PR should include:

- Linked issue
- Scope summary (what changed and why)
- Validation details (how reviewed/checked)
- Risks or follow-up tasks

Keep PRs small and reviewable whenever possible.

## Review Expectations

- Minimum one reviewer
- Reviewer should validate assumptions, not only syntax/style
- Merge only after comments are addressed or resolved

## Project Documentation Standards

- Store phase plans in `docs/`
- Keep report artifacts in `reports/`
- Keep data assumptions explicit and versioned
- Never commit sensitive credentials

## Collaboration Norms

- If scope changes, update the issue before continuing
- If blocked, post blocker and proposed workaround in issue/PR
- If assumptions are uncertain, document them explicitly
