---
name: cuda-preservation
description: Use when work touches CUDA, GPU, Triton, kernels, NVIDIA, PyTorch CUDA, or acceleration paths.
---

Fix CUDA/GPU paths directly. Do not convert GPU training to processor-primary execution, turn CUDA off, force processor tensors, or bypass kernels unless explicitly approved. Run GPU/CUDA-specific tests when present. Always run just ai-guard.
