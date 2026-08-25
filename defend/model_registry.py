"""
Model Registry — Pure artifact loader & in-memory cache for runtime inference.

Invariants:
- ModelRegistry is strictly an artifact loader and runtime cache.
- It NEVER trains or fits models. Training is owned by defend.training.
"""

import os
from typing import Any, Optional

import lightgbm as lgb
from defend.gnn import MerchantGNN, load_gnn_model
from defend.lightgbm_baseline import load_model as load_lgb_booster

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "models")
DEFAULT_LIGHTGBM_PATH = os.path.join(_MODELS_DIR, "lightgbm.txt")
DEFAULT_GNN_PATH = os.path.join(_MODELS_DIR, "gnn.pt")


class ModelArtifactNotFoundError(FileNotFoundError):
    """Raised when a required model artifact is missing from disk."""
    pass


class ModelRegistry:
    _cache: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, model: Any) -> None:
        """Explicitly register an in-memory model (useful for unit testing)."""
        cls._cache[name] = model

    @classmethod
    def load_lightgbm(cls, path: str = DEFAULT_LIGHTGBM_PATH) -> Any:
        cache_key = f"lightgbm:{path}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        if not os.path.exists(path):
            raise ModelArtifactNotFoundError(
                f"LightGBM model artifact not found at '{path}'. "
                "Run offline training via 'python3 -m defend.training.pipeline' first."
            )

        model = load_lgb_booster(path)
        cls._cache[cache_key] = model
        return model

    @classmethod
    def load_gnn(cls, path: str = DEFAULT_GNN_PATH, in_dim: int = 4) -> MerchantGNN:
        cache_key = f"gnn:{path}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        if not os.path.exists(path):
            raise ModelArtifactNotFoundError(
                f"GNN model artifact not found at '{path}'. "
                "Run offline training via 'python3 -m defend.training.pipeline' first."
            )

        model = load_gnn_model(path, in_dim=in_dim)
        cls._cache[cache_key] = model
        return model

    @classmethod
    def get_lightgbm(cls, path: str = DEFAULT_LIGHTGBM_PATH) -> Optional[Any]:
        try:
            return cls.load_lightgbm(path)
        except (ModelArtifactNotFoundError, FileNotFoundError):
            return None

    @classmethod
    def get_gnn(cls, path: str = DEFAULT_GNN_PATH, in_dim: int = 4) -> Optional[MerchantGNN]:
        try:
            return cls.load_gnn(path, in_dim=in_dim)
        except (ModelArtifactNotFoundError, FileNotFoundError):
            return None

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()
