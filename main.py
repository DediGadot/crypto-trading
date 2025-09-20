#!/usr/bin/env python3
"""
SAFLA Cryptocurrency Trading System
Self-Aware Feedback Loop Algorithm for Adaptive Trading

Main entry point for the SAFLA trading system.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run CLI
from safla_trading.cli import cli

if __name__ == '__main__':
    cli()