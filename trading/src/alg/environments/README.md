# Trading Environments

This directory contains the trading environment implementations for the DRL trading system.

## Architecture Overview

The environment system uses an inheritance hierarchy to separate concerns between fast training and accurate evaluation:

```
BaseTradingEnv (ABC)
├── FastTrainingEnv     - Optimized for training speed (target: 10,000 iter/sec)
└── StatefulTradingEnv  - Full state tracking for backtesting/trading
    └── TradingEnv      - Backward compatibility wrapper
```

## Environment Types

### BaseTradingEnv
Abstract base class containing common functionality:
- Data initialization and management
- Observation space definition
- Action space definition
- Common helper methods for observation building

### FastTrainingEnv
**Purpose**: High-speed training environment with minimal state tracking

**Features**:
- Constant-time operations (O(1) complexity)
- No position history tracking
- No complex reward calculations (sharpe ratio, sortino, etc.)
- Pre-computed price matrices for fast lookups
- Simple portfolio value-based rewards
- Minimal memory footprint

**Use Cases**:
- Training DRL agents
- Hyperparameter tuning
- Quick experiments

**Performance Target**: 10,000 iterations per second

**State Tracked**:
- Current cash balance
- Current holdings (shares per symbol)
- Current prices (pre-computed)

**Reward Signal**:
- Simple percentage change in portfolio value
- Normalized with tanh function

### StatefulTradingEnv
**Purpose**: Realistic environment with full portfolio management

**Features**:
- Complete position tracking and history
- Trade constraint enforcement (cash limits, position limits)
- Detailed statistics tracking (returns, cumulative returns, net value)
- Complex reward functions (sharpe ratio, sortino ratio, calmar ratio)
- Realistic order execution through Portfolio and PositionManager
- Full vectorbt integration for analysis

**Use Cases**:
- Backtesting trained models
- Paper trading
- Performance evaluation
- Production trading (with LivePositionManager)

**State Tracked**:
- Full portfolio state through Portfolio class
- Position history through PositionManager
- Trade statistics (returns, cumulative returns, sharpe ratio)
- Complete order and trade history

**Reward Signal**:
- Configurable reward functions (see reward_functions/)
- Supports sharpe ratio, sortino ratio, calmar ratio, profit maximization

### TradingEnv
**Purpose**: Backward compatibility wrapper

This class inherits from `StatefulTradingEnv` and provides the same interface as the original implementation. Existing code using `TradingEnv` will continue to work without modifications.

## Usage Examples

### Training with FastTrainingEnv
```python
from trading.src.alg.environments.fast_training_env import FastTrainingEnv

# Create fast training environment
train_env = FastTrainingEnv(
    data=train_data,
    cfg=stock_env_config,
    features=feature_list,
)

# Train agent (will be much faster)
agent = Agent(agent_config, train_env)
agent.learn()
```

### Backtesting with StatefulTradingEnv
```python
from trading.src.alg.environments.stateful_trading_env import StatefulTradingEnv

# Create stateful environment for accurate evaluation
test_env = StatefulTradingEnv(
    data=test_data,
    cfg=stock_env_config,
    features=feature_list,
)

# Run backtest with full state tracking
backtest = BackTesting(model, test_env, backtest_config)
portfolio = backtest.run()
portfolio.analysis()
```

### Using Original TradingEnv (Backward Compatible)
```python
from trading.src.alg.environments.trading_environment import TradingEnv

# Works exactly as before
env = TradingEnv(data, cfg, features)
# ... existing code ...
```

## Performance Considerations

### FastTrainingEnv Optimizations
1. **Pre-computed Price Matrix**: All prices stored in numpy array for O(1) access
2. **Vectorized Operations**: All computations use numpy vectorization
3. **Minimal State**: Only tracks current holdings and cash (no history)
4. **Simple Rewards**: Direct portfolio value calculation without statistics
5. **No Position Tracking**: No individual position objects or queues

### StatefulTradingEnv Trade-offs
1. **Complete State**: Maintains full history for accurate evaluation
2. **Constraint Enforcement**: Validates all trades against portfolio rules
3. **Complex Metrics**: Calculates sharpe ratio, sortino ratio, etc.
4. **Slower Performance**: ~150-500 iterations per second (but accurate)

## CLI Integration

The CLI commands automatically use the appropriate environment:

- `rr_trading alg train`: Uses **FastTrainingEnv** for speed
- `rr_trading alg backtest`: Uses **StatefulTradingEnv** for accuracy
- `rr_trading alg paper-trade`: Uses **StatefulTradingEnv** with live data

## Benchmarking

Run the performance benchmark to verify improvements:

```bash
python -m trading.test.alg.environments.benchmark_performance
```

Expected results:
- FastTrainingEnv: ~10,000 iterations/second
- StatefulTradingEnv: ~150-500 iterations/second
- Speedup: ~20-50x faster training

## Reward Functions

### For Fast Training
- **fast_profit_reward**: Ultra-fast, tracks only portfolio value changes
- **simple_momentum_reward**: Fast percentage-based returns

### For Evaluation
- **basic_profit_max**: Standard profit maximization
- **sharpe_ratio**: Risk-adjusted returns
- **sortino_ratio**: Downside risk focus
- **calmar_ratio**: Max drawdown focus

Configure in your `stock_env.json`:
```json
{
  "reward_config": {
    "type": "fast_profit_reward",
    "reward_scaling": 10000.0
  }
}
```

## Future Improvements

Possible enhancements:
1. GPU-accelerated computations for even faster training
2. Distributed training across multiple environments
3. Adaptive reward scaling based on market conditions
4. Multi-objective reward functions combining profit and risk
5. Caching mechanisms for frequently accessed data
