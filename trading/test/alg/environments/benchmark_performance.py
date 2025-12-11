"""
Performance benchmark script to compare FastTrainingEnv vs StatefulTradingEnv.
Target: FastTrainingEnv should achieve 10,000 iterations per second.
"""
import time
from pathlib import Path

import numpy as np

from trading.cli.alg.config import RRConfig
from trading.src.alg.data_process.data_loader import DataLoader
from trading.src.alg.environments.fast_training_env import FastTrainingEnv
from trading.src.alg.environments.stateful_trading_env import StatefulTradingEnv

CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"

# Performance target for FastTrainingEnv
TARGET_STEPS_PER_SEC = 10000


def benchmark_environment(env_class, env_name, num_episodes=5, num_steps_per_episode=100):
    """Benchmark an environment over multiple episodes."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {env_name}")
    print(f"{'='*60}")
    
    with Path.open(Path(CONFIG_DIR / "generic_alg.json")) as f:
        alg_config = RRConfig.model_validate_json(f.read())
    
    data_loader = DataLoader(
        data_config=alg_config.data_config, feature_config=alg_config.feature_config
    )
    
    env = env_class(
        data=data_loader.df,
        cfg=alg_config.stock_env,
        features=alg_config.feature_config.features,
    )
    
    total_steps = 0
    total_time = 0.0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_steps = 0
        
        start_time = time.perf_counter()
        
        for _ in range(num_steps_per_episode):
            # Random action
            action = np.random.uniform(-1, 1, env.stock_dimension)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_steps += 1
            
            if terminated or truncated:
                break
        
        episode_time = time.perf_counter() - start_time
        total_steps += episode_steps
        total_time += episode_time
        
        steps_per_sec = episode_steps / episode_time if episode_time > 0 else 0
        print(f"Episode {episode + 1}: {episode_steps} steps in {episode_time:.3f}s "
              f"({steps_per_sec:.1f} steps/sec)")
    
    avg_steps_per_sec = total_steps / total_time if total_time > 0 else 0
    
    print(f"\n{env_name} Summary:")
    print(f"  Total steps: {total_steps}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Average speed: {avg_steps_per_sec:.1f} steps/sec")
    print(f"  Target: {TARGET_STEPS_PER_SEC:,} steps/sec")
    print(f"  Achievement: {(avg_steps_per_sec / TARGET_STEPS_PER_SEC) * 100:.1f}% of target")
    
    return avg_steps_per_sec


def main():
    """Run benchmarks and compare performance."""
    print("\n" + "="*60)
    print("DRL Trading Environment Performance Benchmark")
    print("="*60)
    
    # Benchmark FastTrainingEnv
    fast_speed = benchmark_environment(
        FastTrainingEnv, 
        "FastTrainingEnv (Optimized)", 
        num_episodes=5,
        num_steps_per_episode=200
    )
    
    # Benchmark StatefulTradingEnv
    stateful_speed = benchmark_environment(
        StatefulTradingEnv, 
        "StatefulTradingEnv (Full State)", 
        num_episodes=5,
        num_steps_per_episode=200
    )
    
    # Compare
    print(f"\n{'='*60}")
    print("Performance Comparison")
    print(f"{'='*60}")
    print(f"FastTrainingEnv: {fast_speed:.1f} steps/sec")
    print(f"StatefulTradingEnv: {stateful_speed:.1f} steps/sec")
    
    if stateful_speed > 0:
        speedup = fast_speed / stateful_speed
        print(f"Speedup: {speedup:.2f}x faster")
    
    if fast_speed >= TARGET_STEPS_PER_SEC:
        print(f"\n✓ Target achieved! FastTrainingEnv >= {TARGET_STEPS_PER_SEC:,} steps/sec")
    else:
        print(f"\n✗ Target not achieved. Need {TARGET_STEPS_PER_SEC - fast_speed:.1f} more steps/sec")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
