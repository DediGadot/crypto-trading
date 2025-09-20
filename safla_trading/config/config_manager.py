"""Configuration Manager for SAFLA Trading System.

This module provides centralized configuration management with validation,
environment variable support, and dynamic updates.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger(__name__)


class MemoryConfig(BaseModel):
    """Memory architecture configuration."""
    vector_memory: Dict[str, Any] = Field(default_factory=dict)
    episodic_memory: Dict[str, Any] = Field(default_factory=dict)
    semantic_memory: Dict[str, Any] = Field(default_factory=dict)
    working_memory: Dict[str, Any] = Field(default_factory=dict)


class NeuralConfig(BaseModel):
    """Neural network configuration."""
    safla: Dict[str, Any] = Field(default_factory=dict)
    feedback_loops: Dict[str, Any] = Field(default_factory=dict)
    self_improvement: Dict[str, Any] = Field(default_factory=dict)


class TradingConfig(BaseModel):
    """Trading configuration."""
    exchanges: list = Field(default_factory=list)
    symbols: list = Field(default_factory=list)
    timeframes: Dict[str, Any] = Field(default_factory=dict)
    data_history: Dict[str, Any] = Field(default_factory=dict)


class RiskConfig(BaseModel):
    """Risk management configuration."""
    portfolio: Dict[str, float] = Field(default_factory=dict)
    stop_loss: Dict[str, Any] = Field(default_factory=dict)
    take_profit: Dict[str, Any] = Field(default_factory=dict)
    drawdown: Dict[str, float] = Field(default_factory=dict)


class SystemConfig(BaseModel):
    """System configuration."""
    name: str = "SAFLA Trading System"
    version: str = "1.0.0"
    log_level: str = "INFO"
    max_memory_usage: int = 8192

    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()


class SAFLAConfig(BaseModel):
    """Main SAFLA configuration model."""
    system: SystemConfig = Field(default_factory=SystemConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    neural: NeuralConfig = Field(default_factory=NeuralConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategies: Dict[str, Any] = Field(default_factory=dict)
    performance: Dict[str, Any] = Field(default_factory=dict)
    database: Dict[str, Any] = Field(default_factory=dict)
    logging: Dict[str, Any] = Field(default_factory=dict)


class ConfigManager:
    """Manages configuration loading, validation, and updates."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.

        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config_path = Path(config_path) if config_path else Path("config.yaml")
        self.config: Optional[SAFLAConfig] = None
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file and environment variables."""
        try:
            # Load base configuration from YAML file
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            else:
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                config_data = {}

            # Override with environment variables
            config_data = self._apply_env_overrides(config_data)

            # Validate and create configuration object
            self.config = SAFLAConfig(**config_data)
            logger.info(f"Configuration loaded successfully from {self.config_path}")

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Use default configuration
            self.config = SAFLAConfig()

    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration.

        Environment variables use the format: SAFLA_SECTION_KEY
        Examples:
            SAFLA_SYSTEM_LOG_LEVEL=DEBUG
            SAFLA_TRADING_EXCHANGES_0_API_KEY=your_key
        """
        for key, value in os.environ.items():
            if key.startswith('SAFLA_'):
                # Parse the environment variable name
                parts = key[6:].lower().split('_')  # Remove 'SAFLA_' prefix

                # Navigate to the correct section in config
                current = config_data
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Set the value, converting to appropriate type
                current[parts[-1]] = self._convert_env_value(value)

        return config_data

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate Python type."""
        # Boolean values
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'

        # Numeric values
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # String value
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key.

        Args:
            key: Configuration key in dot notation (e.g., 'memory.vector_memory.dimension')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if not self.config:
            return default

        try:
            parts = key.split('.')
            current = self.config.dict()

            for part in parts:
                current = current[part]

            return current
        except (KeyError, TypeError):
            return default

    def update(self, key: str, value: Any) -> None:
        """Update configuration value by dot notation key.

        Args:
            key: Configuration key in dot notation
            value: New value to set
        """
        if not self.config:
            return

        try:
            parts = key.split('.')
            config_dict = self.config.dict()
            current = config_dict

            # Navigate to parent of target key
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set the value
            current[parts[-1]] = value

            # Recreate config object with validation
            self.config = SAFLAConfig(**config_dict)
            logger.info(f"Configuration updated: {key} = {value}")

        except Exception as e:
            logger.error(f"Failed to update configuration {key}: {e}")

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to file.

        Args:
            path: Path to save configuration. If None, uses current config_path.
        """
        if not self.config:
            return

        save_path = Path(path) if path else self.config_path

        try:
            with open(save_path, 'w') as f:
                yaml.dump(self.config.dict(), f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    def reload(self) -> None:
        """Reload configuration from file."""
        logger.info("Reloading configuration...")
        self._load_config()

    @property
    def memory_config(self) -> MemoryConfig:
        """Get memory configuration."""
        return self.config.memory if self.config else MemoryConfig()

    @property
    def neural_config(self) -> NeuralConfig:
        """Get neural network configuration."""
        return self.config.neural if self.config else NeuralConfig()

    @property
    def trading_config(self) -> TradingConfig:
        """Get trading configuration."""
        return self.config.trading if self.config else TradingConfig()

    @property
    def risk_config(self) -> RiskConfig:
        """Get risk management configuration."""
        return self.config.risk if self.config else RiskConfig()


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager