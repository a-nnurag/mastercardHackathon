"""
Minimal Detection Context abstraction.

Provides a clean container for runtime dependencies (pre-loaded models, graph data,
and runtime configuration) to avoid global variables and ad-hoc constructor hacks across
signal detectors.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DetectionContext:
    model_registry: Any = None
    models: dict[str, Any] = field(default_factory=dict)
    graph: Any = None
    tabular_model: Any = None
    feature_row: Any = None
    gnn_merchant_lookup: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_model(self, name: str, default: Any = None) -> Any:
        if name in self.models:
            return self.models[name]
        if self.model_registry is not None:
            loader = getattr(self.model_registry, f"get_{name}", None) or getattr(self.model_registry, "load", None)
            if callable(loader):
                try:
                    model = loader(name)
                    self.models[name] = model
                    return model
                except Exception:
                    pass
        return default

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
