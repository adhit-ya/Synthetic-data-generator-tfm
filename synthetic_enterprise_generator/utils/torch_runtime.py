"""Safe PyTorch runtime import and CUDA checks."""

from __future__ import annotations

from typing import Any

try:
    import torch as _torch
except (ModuleNotFoundError, ImportError, OSError) as exc:  # pragma: no cover - env-specific.
    torch: Any = None
    TORCH_IMPORT_ERROR: BaseException | None = exc
else:
    torch = _torch
    TORCH_IMPORT_ERROR = None


def torch_unavailable_message() -> str:
    if TORCH_IMPORT_ERROR is None:
        return "PyTorch is not available."
    return (
        "PyTorch could not be imported. This usually means the installed torch "
        "wheel does not match the server CUDA/NCCL runtime. Original error: "
        f"{type(TORCH_IMPORT_ERROR).__name__}: {TORCH_IMPORT_ERROR}"
    )


def require_torch(context: str = "this operation") -> Any:
    if torch is None:
        raise RuntimeError(f"{context} requires PyTorch. {torch_unavailable_message()}") from None
    return torch


def require_cuda_runtime() -> None:
    """Raise a clear error unless CUDA-enabled PyTorch can see a GPU."""

    runtime = require_torch("compute_device='cuda'")
    if not runtime.cuda.is_available():
        raise RuntimeError(
            "compute_device='cuda' was requested, but CUDA-enabled PyTorch and "
            "an available NVIDIA GPU were not detected."
        )
