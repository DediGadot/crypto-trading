#!/usr/bin/env python3
"""
Quick test to verify critical fixes work
"""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

async def main():
    # Just run the enhanced demo with fixes
    from demo_enhanced_trading_system import EnhancedTradingDemo

    try:
        demo = EnhancedTradingDemo()

        # Run individual components to see what works
        print("🚀 TESTING FIXES")
        print("=" * 60)

        # Test exchange connectivity first
        await demo.demo_exchange_connectivity()

        # Test GBDT with fix
        await demo.demo_gbdt_models()

        # Test backtesting with improved strategy
        await demo.demo_backtesting_engine()

        print("\n✅ Tests completed - check output above for results")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())