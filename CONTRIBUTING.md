# Contributing to Quantum Temple

Thanks for your interest!

## Quick Start
- Fork and clone
- Create a feature branch: `git switch -c feat/my-thing`
- Install deps: `pip install -r requirements.txt`
- Run tests: `pytest -q`

## Code Style
- Python 3.11+. Lint with `ruff`.
- Conventional Commits for messages (see `.github/COMMIT_CONVENTIONS.md`).

## Tests
- Put tests under `tests/`. Cover:
  - agents/ archetype policies
  - core/ metrics/ ledger integrity
  - runtime/ engine steps & hooks

## Security / Secrets
- Never commit secrets. Use `.env.secrets` (gitignored) or Vault/SOPS.
- CI runs `gitleaks` to detect secret leaks.

## Benchmarks
- Reproducible runs via `orchestrator/run_benchmark.py`.
- Commit meta + raw logs in `data/benchmarks/<ts>/`.

## Pull Requests
- Ensure tests pass in CI.
- Fill PR template with motivation, risks, and rollout plan.
