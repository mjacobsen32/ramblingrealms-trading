# Fast Training Guide

This guide explains how to use the new FastTrainingEnv to achieve 10,000+ iterations per second during DRL training.

## Overview

The DRL training system has been refactored to separate concerns between fast training and accurate evaluation:

- **FastTrainingEnv**: Optimized for speed (~10,000 iter/sec target)
- **StatefulTradingEnv**: Full state tracking for realistic backtesting (~150-500 iter/sec)
- **TradingEnv**: Backward compatible wrapper (uses StatefulTradingEnv)

## Quick Start

### 1. Training with FastTrainingEnv

Use the pre-configured fast training setup:

```bash
rr_trading alg train --config trading/configs/fast_training_ppo.json
```

This automatically:
- Uses `FastTrainingEnv` for training
- Uses `fast_profit_reward` for ultra-fast reward calculation
- Disables history tracking
- Pre-computes price and feature matrices
- Uses optimized hyperparameters

### 2. Backtesting the Trained Model

After training, evaluate with full state tracking:

```bash
rr_trading alg backtest --config trading/configs/fast_training_ppo.json
```

This automatically:
- Uses `StatefulTradingEnv` for evaluation
- Enforces all trading constraints
- Tracks full portfolio history
- Generates detailed analysis and plots

## Architecture

### FastTrainingEnv Optimizations

1. **Pre-computed Matrices**
   - Price matrix: O(1) price lookups
   - Feature matrix: O(1) feature access
   - No DataFrame slicing during training

2. **Minimal State**
   - Cash balance (scalar)
   - Holdings per symbol (1D array)
   - No position objects
   - No history tracking

3. **Simple Rewards**
   - Direct portfolio value calculation
   - No complex metrics (sharpe, sortino, etc.)
   - Constant time computation

4. **Vectorized Operations**
   - All numpy operations
   - No Python loops
   - Efficient memory access

### StatefulTradingEnv Features

1. **Complete State Tracking**
   - Full position history
   - Trade statistics
   - Returns and cumulative returns

2. **Constraint Enforcement**
   - Cash limits
   - Position limits
   - Trade size limits

3. **Complex Rewards**
   - Sharpe ratio
   - Sortino ratio
   - Calmar ratio
   - Profit maximization

## Configuration

### Reward Functions

Choose the right reward function for your use case:

#### For Training (Fast)
```json
{
  "reward_config": {
    "type": "fast_profit_reward",
    "reward_scaling": 10000.0
  }
}
```

or

```json
{
  "reward_config": {
    "type": "simple_momentum_reward",
    "reward_scaling": 1.0
  }
}
```

#### For Evaluation (Accurate)
```json
{
  "reward_config": {
    "type": "sharpe_ratio",
    "reward_scaling": 1e6,
    "kwargs": {
      "risk_free_rate": 0.02
    }
  }
}
```

### Portfolio Config for Fast Training

Disable features that slow down training:

```json
{
  "portfolio_config": {
    "initial_cash": 100000,
    "maintain_history": false,
    "buy_cost_pct": 0.0,
    "sell_cost_pct": 0.0,
    "trade_mode": "cont",
    "action_threshold": 0.1,
    "trade_limit_percent": 0.1
  }
}
```

### Agent Hyperparameters

Optimized PPO settings for fast training:

```json
{
  "algo": "ppo",
  "total_timesteps": 1e5,
  "kwargs": {
    "policy": "MlpPolicy",
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "ent_coef": 0.01,
    "policy_kwargs": {
      "net_arch": [{"pi": [256, 256], "vf": [256, 256]}]
    }
  }
}
```

## Performance Benchmarking

Run the benchmark to measure performance:

```bash
cd /home/runner/work/ramblingrealms-trading/ramblingrealms-trading
python -m trading.test.alg.environments.benchmark_performance
```

Expected results:
- **FastTrainingEnv**: 8,000-12,000 steps/sec
- **StatefulTradingEnv**: 150-500 steps/sec
- **Speedup**: 20-50x faster

## Workflow

### Development Cycle

1. **Fast Training**
   ```bash
   # Train quickly with FastTrainingEnv
   rr_trading alg train --config fast_training_ppo.json
   ```

2. **Quick Evaluation**
   ```bash
   # Backtest with StatefulTradingEnv
   rr_trading alg backtest --config fast_training_ppo.json
   ```

3. **Iteration**
   - Adjust hyperparameters
   - Try different reward functions
   - Modify network architecture
   - Repeat steps 1-2

4. **Production**
   ```bash
   # Paper trade with full constraints
   rr_trading paper_trade --config trade_config.json
   ```

## Migration Guide

### Existing Code

No changes needed! The original `TradingEnv` class still works:

```python
from trading.src.alg.environments.trading_environment import TradingEnv

# This still works exactly as before
env = TradingEnv(data, cfg, features)
```

### New Code

Use the specialized environments directly:

```python
from trading.src.alg.environments.fast_training_env import FastTrainingEnv
from trading.src.alg.environments.stateful_trading_env import StatefulTradingEnv

# For training
train_env = FastTrainingEnv(train_data, cfg, features)

# For evaluation
test_env = StatefulTradingEnv(test_data, cfg, features)
```

## Troubleshooting

### Training is still slow

1. Check you're using `FastTrainingEnv` (not `TradingEnv`)
2. Verify reward function is `fast_profit_reward` or `simple_momentum_reward`
3. Ensure `maintain_history: false` in portfolio config
4. Check that `total_timesteps` is reasonable

### Backtest results look wrong

1. Ensure you're using `StatefulTradingEnv` for backtesting
2. Verify `maintain_history: true` for accurate tracking
3. Check that portfolio constraints are properly configured
4. Review the generated plots and CSV files

### Model not learning

1. Try different reward scaling values
2. Adjust learning rate
3. Increase `n_steps` or `batch_size`
4. Try `simple_momentum_reward` instead of `fast_profit_reward`
5. Check that features are properly normalized

## Advanced Topics

### Custom Reward Functions

Create your own fast reward function:

```python
from trading.src.alg.environments.reward_functions.base_reward_function import RewardFunction

class MyFastReward(RewardFunction):
    def compute_reward(self, pf, df, realized_profit):
        # Your fast reward logic here
        # Keep it simple and stateless!
        return reward
```

### GPU Acceleration

For even faster training, consider:

1. Use GPU-accelerated PyTorch:
   ```json
   {
     "kwargs": {
       "device": "cuda"
     }
   }
   ```

2. Batch multiple environments:
   ```python
   from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
   
   envs = [lambda: FastTrainingEnv(data, cfg, features) for _ in range(8)]
   vec_env = SubprocVecEnv(envs)
   ```

### Distributed Training

Scale across multiple machines:

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

# Create multiple environment processes
envs = [lambda: FastTrainingEnv(data, cfg, features) for _ in range(16)]
vec_env = SubprocVecEnv(envs)

# Train with parallel environments
model = PPO("MlpPolicy", vec_env)
model.learn(total_timesteps=1e6)
```

## Best Practices

1. **Always train with FastTrainingEnv**
   - Faster iteration
   - More experiments in less time
   - Better hyperparameter tuning

2. **Always backtest with StatefulTradingEnv**
   - Realistic evaluation
   - Proper constraint enforcement
   - Accurate performance metrics

3. **Use appropriate reward functions**
   - Training: `fast_profit_reward`
   - Evaluation: `sharpe_ratio` or `sortino_ratio`

4. **Monitor training progress**
   - Check tensorboard logs
   - Watch for convergence
   - Validate on test set frequently

5. **Version control your configs**
   - Track what works
   - Reproduce experiments
   - Share successful setups

## Support

For issues or questions:
- Check the [README](trading/src/alg/environments/README.md)
- Review example configs in `trading/configs/`
- Run the benchmark script
- Check existing tests in `trading/test/alg/environments/`
