"""
LINUS-APPROVED TRADING SYSTEM
A trading system that actually works, not academic theory
"""

from .config.config_loader import get_config
from .simulator import TradingSimulator, RiskManager
from .data_feed import BinanceDataFeed
from .strategies import SMAStrategy
from .logging_system import TradeLogger

__version__ = "1.0.0"
__all__ = [
    'get_config',
    'TradingSimulator',
    'RiskManager',
    'BinanceDataFeed',
    'SMAStrategy',
    'TradeLogger'
]