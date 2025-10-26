# Conventional Commits (quantum-temple)

Format: `<type>(scope)!: short summary`

Types:
- feat, fix, perf, refactor, test, docs, chore, build, ci, style, revert

Rules:
- Use present tense, imperative mood: "add", "fix", "refactor"
- Keep subject ≤ 72 chars; wrap body at 100 cols
- Use `BREAKING CHANGE:` in footer when applicable
- Reference issues like `Fixes #123`

Examples:
- feat(runtime): add sigma_Q PID controller with variance target
- fix(core): CPTP dephasing channel for SigilFactory
- perf(qnn): use expm_multiply to avoid dense U materialization
- test(sync): add PLV edge-case tests for phase lock step
