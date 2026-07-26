# Rambling Realms Trading Platform - Agent Guidelines

This is a Python 3.12+ quantitative trading platform for training RL agents (PPO, DDPG, etc.) on multi-ticker OHLCV data with fundamental EDGAR features. Goal is long-term trend discovery, not HFT.

## Project Structure
- `trading/src/alg/environments/` - RL envs: `BaseTradingEnv` (ABC), `FastTrainingEnv` (vectorized, 10k iter/s, symbol shuffling), `StatefulTradingEnv` (full Portfolio tracking), `TradingEnv` (legacy compat)
- `trading/src/alg/environments/reward_functions/` - configurable reward factory (`reward_factory_method`)
- `trading/src/alg/data_process/` - data loading, train/test split
- `trading/src/alg/agents/` - SB3 wrappers, LR schedules
- `trading/src/alg/callbacks/` - TensorBoard telemetry
- `trading/src/alg/backtest/` - backtesting runner
- `trading/src/portfolio/` - `Portfolio` + `PositionManager` (vectorbt-backed), trade rules enforcement
- `trading/src/features/` - Feature registry (`Feature._registry`, factory via `type` field). Implementations: `Candle`, `MovingWindow`, `RSI`, `MACD`, `BollingerBands`, `ATR`, `MSTD`, `OBV`, `Stochastic`, `Turbulence`, `Edgar*` family
- `trading/src/data/` - `EDGARClient` + `EdgarToolsClient` (SEC rate-limit 10 req/s, caching), data blending
- `trading/src/trade/` - Alpaca paper/live trading clients
- `trading/src/user_cache/` - user secrets/cache (polygon, alpaca keys via env `RR_TRADING_USER_CACHE_PATH` in tests)
- `trading/cli/` - Typer CLI: `rr_trading data`, `alg train/backtest/analysis`, `paper_trade`/`live_trade`/`push_backtest`
- `trading/configs/` - JSON configs: `agents/`, `data/`, `envs/`, `features/`, `trading/` + composite configs like `fast_training_ppo.json`
- `trading/test/` - pytest suite; live EDGAR tests gated by env var
- `docs/` - Sphinx (furo + autoapi)

## Package Management & Tooling
- **Poetry**: `pyproject.toml` is source of truth (poetry-core). Run `poetry install --with dev` for dev deps
- **Python**: `>=3.12,<4.0`. Black target `py312`
- **Entry points**: `rr_trading`, `benchmark-performance`
- **CI** (`.github/workflows/ci.yml`): pre-commit (black, isort, mypy, poetry-check + export) + `coverage run -m pytest -m "not benchmark"` with 90% threshold (`tool.coverage.report.fail_under=90`)
- **Lint/format**: black 25.1, isort 6.0.1 (profile black), ruff 0.11.13, mypy 1.16 (+ types-requests)
- **Pre-commit**: must run `pre-commit run --all-files` before commit

## Key Conventions
- Feature addition: subclass `Feature`, set `TYPE: ClassVar[FeatureType] = FeatureType.X`, implement `to_df(self, df, data)` and `get_feature_names()`. Auto-registered via `__init_subclass__`. Add enum to `FeatureType`.
- Env hierarchy: inherit from `BaseTradingEnv`; override `_get_observation` and `step`. Training uses `FastTrainingEnv` (minimal state, precomputed price_matrix/feature_matrix, symbol permutation for anti-memorization). Backtest/paper uses `StatefulTradingEnv` (full Portfolio, profit tracking).
- Portfolio trade flow: `scale_actions` (signal -> shares) -> `enforce_trade_rules` (cash/hmax/trade_limit) -> `PositionManager.step`
- Config system: Pydantic `RRConfig` with `ProjectPath` templating (`{PROJECT_ROOT}`, `{OUT_DIR}`, `{BACKTEST_DIR}`, `{VERSION}`, `{TIMESTAMP}`, `{UUID}`). JSON files loaded via `ProjectPath.model_validate` + `model_validate_json`.
- Cache: `data/` gitignored; EDGAR and market data cached locally. Tests use small parquets under `trading/test/data/` (!-preserved in .gitignore)
- Logging: `trading.cli.utils.init_logger` + `init_file_logger`; use `logging` module, not prints.

## Build, Test, Verification
```bash
poetry install --with dev
poetry run pre-commit run --all-files   # lint gate
poetry run pytest -m "not benchmark" -xvs
poetry run coverage run -m pytest -m "not benchmark" && poetry run coverage report --fail-under=90
poetry run pytest trading/test/alg/environments/test_trading_environment.py -xvs  # focused
```
For GPU path: optional dep group `gpu` (cudf-cu12). Do not require in CI.

## Gotchas
- `PositionManager` is stateful; reset between episodes via `_reset_internal_states`.
- `observation_timestamp` must be set in `init_data` and `_reset_internal_states` - guard against None.
- EDGAR clients respect SEC 10 req/s; use `_respect_rate_limit` and cache_dir with TTL. Do not hit live SEC in unit tests unless `RUN_LIVE_EDGAR_TESTS=1`.
- `Portfolio.as_vbt_pf()` pivots data; ensure MultiIndex `(timestamp,symbol)` with `price` column.
- `TradeMode.CONTINUOUS` vs `DISCRETE` changes action_space dtype (float32 vs int32) and sell/buy logic.
- Do not commit `data/`, `models/`, `logs/`, `out/`, `dist/`, `.project`, `trading/cli/configuration/prod.yml` (all gitignored).
- Coverage exclusions: `trading/cli/rr_trading.py`, `**/polygon_io.py`, tests.

## PR Guidelines
- Keep edits minimal, reuse existing utils. Prefer editing over new files.
- Update `trading/configs/` JSONs if adding env/agent features.
- Run mypy and ruff; ensure black/isort pass.
- Keep `fail_under=90` - add tests for new code paths.
