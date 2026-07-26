---
description: Runs lint gate (black, isort, ruff, mypy, poetry-check) via pre-commit
agent: build
---

Run the full lint gate for this repo and fix any auto-fixable issues.

Steps:
1. Run `poetry run pre-commit run --all-files`
2. If black/isort fail, run fixes and re-run
3. If mypy fails, report file:line with suggested fixes (do not suppress without reason)
4. If ruff fails, fix
5. Final report: pass/fail for each hook

Constraints:
- project requires Python 3.12+, black target py312, isort profile black
- Use `poetry run` prefix for all tooling
- Do not modify poetry.lock manually - use poetry install if needed
