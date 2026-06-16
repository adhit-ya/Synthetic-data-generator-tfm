"""Adapter around TabPFN ecosystem synthetic priors.

TabPFN itself is primarily an inference-time foundation model, while TabPFGen
and parts of tabpfn-extensions expose prior/data generation utilities depending
on installed versions. Their public APIs have changed over time, so this module
uses defensive dynamic discovery: known generator function names are attempted
first, and a deterministic scikit-learn fallback is used only when no installed
ecosystem generator can be called.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np
from sklearn.datasets import make_classification, make_regression

from synthetic_enterprise_generator.utils.torch_runtime import (
    require_cuda_runtime,
    torch,
    torch_unavailable_message,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class TabPFNAdapterResult:
    """Container returned by the ecosystem adapter."""

    X: np.ndarray
    y: np.ndarray
    source: str


class TabPFNEcosystemGenerator:
    """Generate base IID-like tables from TabPFN ecosystem priors when present.

    The enterprise structure is added in later stages. This class deliberately
    stays focused on the base feature/target prior so it can be swapped for a
    concrete TabPFGen API in production without touching downstream modules.
    """

    candidate_modules = (
        "tabpfgen",
        "tabpfgen.generator",
        "tabpfn_extensions",
        "tabpfn_extensions.priors",
        "tabpfn_extensions.synthetic",
    )
    candidate_functions = (
        "generate_dataset",
        "generate_data",
        "sample_dataset",
        "sample_tabular_data",
        "generate_classification_dataset",
        "generate_regression_dataset",
    )

    def __init__(
        self,
        require_ecosystem: bool = False,
        compute_device: str = "auto",
        max_ecosystem_rows: int = 10_000,
    ) -> None:
        self.require_ecosystem = require_ecosystem
        self.compute_device = self._resolve_device(compute_device)
        self.max_ecosystem_rows = max(1, int(max_ecosystem_rows))
        self.available_components = self._detect_components()
        LOGGER.info("Base prior compute device: %s", self.compute_device)

    def generate(
        self,
        task: str,
        n_rows: int,
        n_features: int,
        n_classes: int,
        rng: np.random.Generator,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> TabPFNAdapterResult:
        """Generate a base matrix and target vector.

        Parameters are intentionally generic because TabPFGen/tabpfn-extension
        versions differ. The dynamic call layer maps these values onto whatever
        keyword names a discovered function accepts.
        """

        ecosystem_result = None
        if self.require_ecosystem or n_rows <= self.max_ecosystem_rows:
            ecosystem_result = self._try_ecosystem_generator(
                task=task,
                n_rows=n_rows,
                n_features=n_features,
                n_classes=n_classes,
                rng=rng,
                progress_callback=progress_callback,
            )
        else:
            LOGGER.info(
                "Skipping TabPFN ecosystem generator for %s rows; the configured "
                "limit is %s. Using the scalable fallback.",
                f"{n_rows:,}",
                f"{self.max_ecosystem_rows:,}",
            )
        if ecosystem_result is not None:
            return ecosystem_result
        if self.require_ecosystem:
            raise RuntimeError(
                "No callable TabPFN/TabPFGen/tabpfn-extensions generator was found. "
                "Install the ecosystem packages or disable require_tabpfn_ecosystem."
            )
        LOGGER.warning(
            "No TabPFN ecosystem generator was selected. Using the scalable %s "
            "base-data fallback; downstream augmentation still runs unchanged.",
            "CUDA" if self.compute_device == "cuda" else "sklearn CPU",
        )
        if self.compute_device == "cuda":
            return self._torch_generate(
                task, n_rows, n_features, n_classes, rng, progress_callback
            )
        return self._fallback_generate(
            task, n_rows, n_features, n_classes, rng, progress_callback
        )

    def _resolve_device(self, requested: str) -> str:
        normalized = str(requested).lower()
        if normalized not in {"auto", "cpu", "cuda"}:
            raise ValueError(
                f"Unsupported compute_device {requested!r}; use auto, cpu, or cuda."
            )
        cuda_available = bool(torch is not None and torch.cuda.is_available())
        if normalized == "cuda" and not cuda_available:
            require_cuda_runtime()
        if normalized == "auto":
            return "cuda" if cuda_available else "cpu"
        return normalized

    def _detect_components(self) -> Tuple[str, ...]:
        found = []
        for module_name in ("tabpfn", "tabpfn_extensions", "tabpfgen"):
            try:
                importlib.import_module(module_name)
                found.append(module_name)
            except Exception:
                continue
        if found:
            LOGGER.info("Detected TabPFN ecosystem components: %s", ", ".join(found))
        else:
            LOGGER.warning("No TabPFN ecosystem packages detected on import path.")
        return tuple(found)

    def _try_ecosystem_generator(
        self,
        task: str,
        n_rows: int,
        n_features: int,
        n_classes: int,
        rng: np.random.Generator,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> Optional[TabPFNAdapterResult]:
        seed = int(rng.integers(0, 2**31 - 1))
        for module_name in self.candidate_modules:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            for function_name in self.candidate_functions:
                function = getattr(module, function_name, None)
                if function is None or not callable(function):
                    continue
                if self.compute_device == "cuda" and not self._supports_device(function):
                    LOGGER.debug(
                        "Skipping %s.%s because it cannot accept an explicit CUDA device.",
                        module_name,
                        function_name,
                    )
                    continue
                try:
                    result = self._call_generator(
                        function=function,
                        task=task,
                        n_rows=n_rows,
                        n_features=n_features,
                        n_classes=n_classes,
                        seed=seed,
                        device=self.compute_device,
                    )
                    parsed = self._parse_result(result)
                    if parsed is not None:
                        if progress_callback is not None:
                            progress_callback(f"{task} rows", int(parsed[0].shape[0]))
                        LOGGER.info(
                            "Generated base table using %s.%s",
                            module_name,
                            function_name,
                        )
                        return TabPFNAdapterResult(
                            X=parsed[0], y=parsed[1], source=f"{module_name}.{function_name}"
                        )
                except Exception as exc:
                    LOGGER.debug(
                        "Skipping %s.%s due to call error: %s",
                        module_name,
                        function_name,
                        exc,
                    )
        return None

    def _call_generator(self, function: Any, **values: Any) -> Any:
        """Call an unknown generator with only the kwargs it supports."""

        signature = inspect.signature(function)
        aliases = {
            "n_rows": ("n_rows", "num_rows", "n_samples", "num_samples"),
            "n_features": ("n_features", "num_features", "num_columns"),
            "n_classes": ("n_classes", "num_classes"),
            "task": ("task", "problem_type"),
            "seed": ("seed", "random_state"),
            "device": ("device", "compute_device", "device_type"),
        }
        kwargs = {}
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
        for canonical, names in aliases.items():
            for name in names:
                if accepts_var_kwargs or name in signature.parameters:
                    kwargs[name] = values[canonical]
                    break
        return function(**kwargs)

    def _parse_result(self, result: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Normalize common generator return shapes into ``(X, y)`` arrays."""

        if isinstance(result, tuple) and len(result) >= 2:
            X = self._as_numpy(result[0])
            y = self._as_numpy(result[1])
            if X.ndim == 2 and len(y) == X.shape[0]:
                return X, y
        if hasattr(result, "X") and hasattr(result, "y"):
            X = self._as_numpy(result.X)
            y = self._as_numpy(result.y)
            if X.ndim == 2 and len(y) == X.shape[0]:
                return X, y
        if hasattr(result, "data") and hasattr(result, "target"):
            X = self._as_numpy(result.data)
            y = self._as_numpy(result.target)
            if X.ndim == 2 and len(y) == X.shape[0]:
                return X, y
        return None

    def _as_numpy(self, value: Any) -> np.ndarray:
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _supports_device(self, function: Any) -> bool:
        signature = inspect.signature(function)
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            return True
        return any(
            name in signature.parameters
            for name in ("device", "compute_device", "device_type")
        )

    def _torch_generate(
        self,
        task: str,
        n_rows: int,
        n_features: int,
        n_classes: int,
        rng: np.random.Generator,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> TabPFNAdapterResult:
        """Generate the scalable base matrix on CUDA when explicitly available."""

        if torch is None:
            raise RuntimeError(
                "PyTorch is required for CUDA base generation. "
                f"{torch_unavailable_message()}"
            )
        seed = int(rng.integers(0, 2**31 - 1))
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        chunk_size = min(n_rows, max(1_024, min(16_384, n_rows // 10 or n_rows)))
        chunks = []
        remaining = int(n_rows)
        while remaining:
            current = min(chunk_size, remaining)
            chunks.append(
                torch.randn(
                    (current, n_features),
                    generator=generator,
                    device="cuda",
                    dtype=torch.float32,
                )
            )
            remaining -= current
            if progress_callback is not None:
                progress_callback(f"{task} rows", current)
        X = torch.cat(chunks, dim=0)
        informative = max(2, min(n_features, int(n_features * 0.65)))
        if task == "classification":
            weights = torch.randn(
                (informative, n_classes),
                generator=generator,
                device="cuda",
            )
            logits = X[:, :informative] @ weights
            if informative >= 2:
                interaction_weights = torch.randn(
                    (n_classes,),
                    generator=generator,
                    device="cuda",
                )
                logits += (
                    X[:, 0] * X[:, 1]
                ).unsqueeze(1) * interaction_weights.unsqueeze(0)
            logits += 0.15 * torch.randn(
                logits.shape,
                generator=generator,
                device="cuda",
            )
            y = torch.argmax(logits, dim=1)
        else:
            weights = torch.randn(
                (informative,),
                generator=generator,
                device="cuda",
            )
            y = X[:, :informative] @ weights
            if informative >= 2:
                y += 0.35 * X[:, 0] * X[:, 1]
            y += 0.15 * torch.randn(
                (n_rows,),
                generator=generator,
                device="cuda",
            )
        LOGGER.info(
            "Generated scalable %s base prior on CUDA: rows=%s features=%s",
            task,
            n_rows,
            n_features,
        )
        return TabPFNAdapterResult(
            X=X.cpu().numpy(),
            y=y.cpu().numpy(),
            source="torch_cuda_fallback",
        )

    def _fallback_generate(
        self,
        task: str,
        n_rows: int,
        n_features: int,
        n_classes: int,
        rng: np.random.Generator,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> TabPFNAdapterResult:
        seed = int(rng.integers(0, 2**31 - 1))
        if task == "classification":
            weights = np.linspace(1.0, 0.35, n_classes)
            weights = (weights / weights.sum()).tolist()
            X, y = make_classification(
                n_samples=n_rows,
                n_features=n_features,
                n_informative=max(2, min(n_features, int(n_features * 0.65))),
                n_redundant=max(0, min(n_features - 2, int(n_features * 0.15))),
                n_repeated=0,
                n_classes=n_classes,
                weights=weights,
                class_sep=1.0,
                flip_y=0.03,
                random_state=seed,
            )
        else:
            X, y = make_regression(
                n_samples=n_rows,
                n_features=n_features,
                n_informative=max(2, int(n_features * 0.65)),
                noise=15.0,
                random_state=seed,
            )
        if progress_callback is not None:
            progress_callback(f"{task} rows", int(n_rows))
        return TabPFNAdapterResult(X=np.asarray(X), y=np.asarray(y), source="sklearn_fallback")
