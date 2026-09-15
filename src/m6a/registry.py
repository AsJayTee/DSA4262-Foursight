"""Name -> implementation lookup for features and models.

Discovery is by directory scan, so a new experiment is a new file and nothing
existing has to be edited. Four people's agents can add modules in parallel
without touching a shared table, which means no merge conflicts.

A module that fails to import is skipped and its error recorded rather than
raised. That is deliberate: models/mil_torch.py must not make torch a hard
dependency of predict.py, which evaluators run on a clean machine. The cost is
that a genuinely broken module goes quiet, so `available()` and the KeyError
message both surface the recorded failures.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable

_PACKAGES = {"features": "m6a.features", "models": "m6a.models"}

_registry: dict[str, dict[str, Any]] = {k: {} for k in _PACKAGES}
_failures: dict[str, dict[str, str]] = {k: {} for k in _PACKAGES}
_discovered: set[str] = set()


def register(kind: str, name: str) -> Callable[[Any], Any]:
    """Decorator: register a class under `name`.

    @register("models", "lightgbm")
    class LightGBMModel: ...
    """
    if kind not in _PACKAGES:
        raise ValueError(f"Unknown kind {kind!r}; expected one of {sorted(_PACKAGES)}")

    def wrap(obj: Any) -> Any:
        if name in _registry[kind]:
            raise ValueError(
                f"{kind}/{name!r} is already registered by "
                f"{_registry[kind][name].__module__}. Pick a different name."
            )
        obj.name = name
        _registry[kind][name] = obj
        return obj

    return wrap


def _discover(kind: str) -> None:
    if kind in _discovered:
        return
    _discovered.add(kind)

    package = importlib.import_module(_PACKAGES[kind])
    for info in pkgutil.iter_modules(package.__path__):
        if info.name.startswith("_") or info.name == "base":
            continue
        try:
            importlib.import_module(f"{_PACKAGES[kind]}.{info.name}")
        except Exception as exc:  # noqa: BLE001 - see module docstring
            _failures[kind][info.name] = f"{type(exc).__name__}: {exc}"


def get(kind: str, name: str) -> Any:
    _discover(kind)
    if name not in _registry[kind]:
        lines = [
            f"No {kind[:-1]} registered as {name!r}.",
            f"Available: {', '.join(sorted(_registry[kind])) or '(none)'}",
        ]
        if _failures[kind]:
            lines.append("Modules that failed to import (so their names are missing):")
            lines += [f"  {m}: {err}" for m, err in sorted(_failures[kind].items())]
        raise KeyError("\n".join(lines))
    return _registry[kind][name]


def available(kind: str) -> list[str]:
    _discover(kind)
    return sorted(_registry[kind])


def failures(kind: str) -> dict[str, str]:
    _discover(kind)
    return dict(_failures[kind])
