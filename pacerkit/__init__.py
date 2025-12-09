"""
PacerKit - PACER Model Merging Framework

Permutation-Aligned Consensus Expert Routing for base-free,
interference-aware model merging in LLMs and Vision Transformers.
"""

__version__ = "0.1.0"

# Lazy imports to avoid requiring torch for config-only usage
def __getattr__(name):
    if name == "PACERMerger":
        from pacerkit.pacer import PACERMerger
        return PACERMerger
    elif name == "PACERConfig":
        from pacerkit.config import PACERConfig
        return PACERConfig
    elif name == "load_config":
        from pacerkit.config import load_config
        return load_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PACERMerger",
    "PACERConfig",
    "load_config",
    "__version__",
]
