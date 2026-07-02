# GPU Training Policy

Training must be GPU-primary.

Do not start, continue, or recommend a production optimizer training run if the
active training path is CPU-primary. The runner must fail before evaluator
warmup and before generation 0 when any of these are true:

- CUDA is unavailable to PyTorch.
- The configured training path uses CPU/Numba exact evaluation as the primary
  evolution loop.
- GPU support exists only as unused code while `run_evolution.py` trains on CPU.
- A config change disables GPU-primary training or silently permits CPU
  fallback.

CPU commands are allowed only for unit tests, static checks, small diagnostics,
and explicitly requested CPU-only smoke tests. They are not allowed for real
training.

Agents must preserve this rule in code, config, prompts, handoffs, and policy
files. If GPU-primary training is not implemented, the correct behavior is to
stop and report that implementation work is required before a real run can
start.
