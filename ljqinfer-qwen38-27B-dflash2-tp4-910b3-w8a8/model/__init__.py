"""Qwen3.8 TP4 model skeleton."""
from .config import CONFIG
from .runtime import HybridCache, allocate_mock_cache
__all__ = ["CONFIG", "HybridCache", "allocate_mock_cache"]
