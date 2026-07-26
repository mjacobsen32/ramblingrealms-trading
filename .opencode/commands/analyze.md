---
description: Comprehensive analysis of env, portfolio, or feature code
agent: build
---

Perform deep analysis of $ARGUMENTS area in this trading platform.

Areas supported: env, portfolio, feature, data, config, trade, backtest, edgar

1. Locate relevant files via Glob `trading/src/**/$ARGUMENTS*.py` and Grep for class names
2. Read key implementation files fully
3. Summarize:
   - Architecture and key abstractions
   - Critical paths (hot loops in step(), to_df())
   - Known gotchas from AGENTS.md that apply
   - Potential improvements or bugs with file:line references
4. Suggest next steps if any

If $ARGUMENTS is empty, default to analyzing the full alg/environments stack.
