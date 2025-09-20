"""
PROPER CONFIG LOADING
Not your nested dictionary hell with magic defaults
"""

import yaml
from typing import Any, Dict
from pathlib import Path
import os


class Config:
    """Config loader that actually works"""

    def __init__(self, config_path: str = "config.yaml"):
        """Load config from file

        Args:
            config_path: Path to config file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        # Validate required sections
        required_sections = [
            'system', 'exchange', 'symbols', 'market_data',
            'simulation', 'strategy', 'risk', 'logging'
        ]

        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required config section: {section}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value using dot notation

        Args:
            key: Config key like 'system.name' or 'risk.stop_loss_pct'
            default: Default value if key not found

        Returns:
            Config value
        """
        keys = key.split('.')
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Config key not found: {key}")

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire config section

        Args:
            section: Section name like 'strategy' or 'risk'

        Returns:
            Config section as dictionary
        """
        if section not in self._config:
            raise KeyError(f"Config section not found: {section}")
        return self._config[section]

    @property
    def system_name(self) -> str:
        return self.get('system.name')

    @property
    def log_level(self) -> str:
        return self.get('system.log_level')

    @property
    def initial_balance(self) -> float:
        return self.get('simulation.initial_balance_usd')

    @property
    def symbols(self) -> list:
        return self.get('symbols.primary')

    @property
    def test_symbol(self) -> str:
        return self.get('symbols.test')[0]

    @property
    def commission_rate(self) -> float:
        return self.get('simulation.commission_rate')

    @property
    def stop_loss_pct(self) -> float:
        return self.get('risk.stop_loss_pct')

    @property
    def take_profit_pct(self) -> float:
        return self.get('risk.take_profit_pct')

    @property
    def max_position_size_usd(self) -> float:
        return self.get('strategy.max_position_size_usd')

    @property
    def position_size_pct(self) -> float:
        return self.get('strategy.position_size_pct')

    @property
    def fast_period(self) -> int:
        return self.get('strategy.fast_period')

    @property
    def slow_period(self) -> int:
        return self.get('strategy.slow_period')

    @property
    def max_daily_trades(self) -> int:
        return self.get('risk.max_daily_trades')

    @property
    def max_open_positions(self) -> int:
        return self.get('risk.max_open_positions')


# Global config instance - loaded once
_config = None

def get_config() -> Config:
    """Get global config instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config

def reload_config() -> Config:
    """Reload config from file"""
    global _config
    _config = Config()
    return _config