"""Unified configuration helpers for the trading system.

Historically the project shipped two different configuration loaders.  For the
purposes of the trimmed-down codebase we expose a single `get_config` function
that returns the simple YAML-backed configuration object defined in
``config_loader``.  The Pydantic ``ConfigManager`` remains available for users
that prefer runtime validation, but it is optional.
"""

from .config_loader import Config, get_config as _basic_get_config, reload_config

try:  # Optional dependency: pydantic
    from .config_manager import ConfigManager  # type: ignore
except Exception:  # pragma: no cover - fallback when pydantic is absent
    ConfigManager = None  # type: ignore


def get_config() -> Config:
    """Return the default YAML-backed configuration instance."""

    return _basic_get_config()


__all__ = ["Config", "get_config", "reload_config", "ConfigManager"]
