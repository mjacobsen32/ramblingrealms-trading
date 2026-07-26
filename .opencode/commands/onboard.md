---
description: Onboard and verify environment setup
agent: build
---

Onboard this repo:

1. Check Python version: `python --version` must be 3.12+
2. Check if poetry is available and run `poetry install --with dev` simulation check (don't run full install unless deps missing)
3. Check existence of test data `trading/test/data/TEST_*.parquet`
4. Run `poetry run pytest -m "not benchmark" -q 2>&1 | tail -20` to verify baseline
5. Run `poetry run pre-commit run --all-files 2>&1 | tail -30` or at least check hooks exist
6. Summarize project structure from AGENTS.md and suggest next commands

Do not modify any source files.
