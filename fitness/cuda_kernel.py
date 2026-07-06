"""CUDA exact fitness kernel wrapper.

Loads the compiled CUDA fitness kernel and exposes `evaluate_batch_cuda`.
The kernel mirrors the Numba `_single_genome` / `_evaluate_batch` logic.
"""
import os
import sys
import warnings
from typing import Any, Optional, Tuple

import numpy as np
import torch

_CUDA_AVAILABLE = torch.cuda.is_available()
_CUDA_EXTENSION = None


def _extension_name() -> str:
    return "charybdis_fitness_cuda"


def _source_path() -> str:
    return os.path.join(os.path.dirname(__file__), "cuda", "fitness_kernel.cu")


def _load_cuda_extension():
    global _CUDA_EXTENSION
    if _CUDA_EXTENSION is not None:
        return _CUDA_EXTENSION
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA is not available")

    # Ninja is installed in the venv; ensure it is on PATH for the build.
    venv_bin = os.path.join(sys.prefix, "bin")
    if os.path.isdir(venv_bin) and venv_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")

    from torch.utils.cpp_extension import load

    source = _source_path()
    if not os.path.exists(source):
        raise FileNotFoundError(f"CUDA kernel source not found: {source}")

    build_dir = os.path.join(os.path.dirname(__file__), "cuda", "build")
    os.makedirs(build_dir, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _CUDA_EXTENSION = load(
            name=_extension_name(),
            sources=[source],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            build_directory=build_dir,
            verbose=False,
        )
    return _CUDA_EXTENSION


def _to_tensor(arr: Any, device: str = "cuda") -> torch.Tensor:
    if isinstance(arr, torch.Tensor):
        return arr.to(device)
    t = torch.from_numpy(np.asarray(arr))
    return t.to(device)


def _to_cpu(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _build_args(arrays: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
    """Convert the precomputed arrays tuple to CUDA tensors."""
    (
        pos_effort,
        pos_layer,
        pos_finger,
        pos_hand,
        pos_is_thumb,
        pos_is_frozen,
        dist,
        trackball_dist,
        pos_x,
        pos_y,
        shortcut_importance,
        shortcut_app,
        shortcut_category,
        shortcut_base,
        shortcut_l0_only,
        shortcut_trackball,
        shortcut_is_mouse,
        shortcut_mouse_button,
        shortcut_preferred_hand,
        shortcut_arrow_type,
        shortcut_raw_completion,
        shortcut_raw_completion_base,
        shortcut_access_target,
        shortcut_access_momentary,
        shortcut_scroll_mode_access,
        shortcut_usage_count,
        app_usage_weight,
        group_matrix,
        sequence_rows,
        app_workflow_rows,
        duplicate_support,
        chain_rows,
        workflow_rows,
        blind_rows,
        reference_genome,
        objective_weights,
        violation_weights,
        scale_factors,
        threshold,
        hard_constraint_indices,
        shortcut_key_group,
        n_key_groups,
        toggle_effort_multiplier,
        log1p_lut,
        pos_effort_waste,
    ) = arrays
    return (
        _to_tensor(pos_effort),
        _to_tensor(pos_layer),
        _to_tensor(pos_finger),
        _to_tensor(pos_hand),
        _to_tensor(pos_is_thumb),
        _to_tensor(pos_is_frozen),
        _to_tensor(dist),
        _to_tensor(trackball_dist),
        _to_tensor(pos_x),
        _to_tensor(pos_y),
        _to_tensor(shortcut_importance),
        _to_tensor(shortcut_app),
        _to_tensor(shortcut_category),
        _to_tensor(shortcut_base),
        _to_tensor(shortcut_l0_only),
        _to_tensor(shortcut_trackball),
        _to_tensor(shortcut_is_mouse),
        _to_tensor(shortcut_mouse_button),
        _to_tensor(shortcut_preferred_hand),
        _to_tensor(shortcut_arrow_type),
        _to_tensor(shortcut_raw_completion),
        _to_tensor(shortcut_raw_completion_base),
        _to_tensor(shortcut_access_target),
        _to_tensor(shortcut_access_momentary),
        _to_tensor(shortcut_scroll_mode_access),
        _to_tensor(shortcut_usage_count),
        _to_tensor(app_usage_weight),
        _to_tensor(group_matrix),
        _to_tensor(sequence_rows),
        _to_tensor(app_workflow_rows),
        _to_tensor(duplicate_support),
        _to_tensor(chain_rows),
        _to_tensor(workflow_rows),
        _to_tensor(blind_rows),
        _to_tensor(reference_genome),
        _to_tensor(objective_weights),
        _to_tensor(violation_weights),
        _to_tensor(scale_factors),
        _to_tensor(threshold),
        _to_tensor(hard_constraint_indices),
        _to_tensor(shortcut_key_group),
        _to_tensor(n_key_groups),
        _to_tensor(toggle_effort_multiplier),
        _to_tensor(log1p_lut),
        _to_tensor(pos_effort_waste),
    )


def build_cuda_args(arrays: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
    """Convert precomputed numpy arrays to CUDA tensors once and cache."""
    n_groups = arrays[27].shape[0]
    args = _build_args(arrays)
    # Replace the n_key_groups tensor with the actual number of protected groups.
    return args[:41] + (_to_tensor(n_groups),) + args[42:]


def evaluate_batch_cuda(
    genomes: np.ndarray,
    arrays: Tuple[Any, ...],
    cuda_args: Optional[Tuple[torch.Tensor, ...]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate a batch of genomes on the CUDA fitness kernel.

    Returns (objectives, constraints) as CPU numpy arrays.
    If `cuda_args` is provided, skips rebuilding the static CUDA tensors.
    """
    ext = _load_cuda_extension()

    genomes = np.asarray(genomes, dtype=np.int32)
    if genomes.ndim == 1:
        genomes = genomes.reshape(1, -1)
    batch = genomes.shape[0]

    n_constr = arrays[39].shape[0]
    objectives = torch.empty((batch, 3), dtype=torch.float32, device="cuda")
    constraints = torch.empty((batch, n_constr), dtype=torch.float32, device="cuda")

    args = cuda_args if cuda_args is not None else build_cuda_args(arrays)

    ext.evaluate_batch(_to_tensor(genomes), objectives, constraints, *args)

    return _to_cpu(objectives), _to_cpu(constraints)


def cuda_available() -> bool:
    return _CUDA_AVAILABLE
