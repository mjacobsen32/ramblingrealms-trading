---
description: Benchmark FastTrainingEnv performance
agent: build
---

Benchmark the FastTrainingEnv to verify 10k iter/s target:

```bash
poetry run pytest trading/test/benchmark/test_benchmark-train-env.py -xvs --benchmark-only 2>&1 | tail -100
```

Also run the dedicated benchmark entry:

```bash
poetry run benchmark-performance 2>&1 | tail -100
```

Report:
- iterations/sec achieved
- Whether target met
- Bottlenecks if below target (profile step(), _get_observation())
