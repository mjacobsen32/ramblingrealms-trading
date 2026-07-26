---
description: Run fast test suite excluding benchmarks
agent: build
---

Run focused verification:

```bash
poetry run pytest -m "not benchmark" -xvs -k "$ARGUMENTS" 2>&1 | head -200
```

If no $ARGUMENTS provided, run full suite:

```bash
poetry run pytest -m "not benchmark" -x 2>&1 | tail -50
```

Always report number passed/failed and highlight first failure with file:line.

If coverage check requested, run:
```bash
poetry run coverage run -m pytest -m "not benchmark" && poetry run coverage report --fail-under=90
```
