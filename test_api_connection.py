#!/usr/bin/env python3
"""
TEST API CONNECTION
Quick test script to verify Binance API credentials
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from safla_trading.connectivity import get_exchange_registry
from safla_trading.logging_system import TradeLogger


async def test_api_connection():
    """Test API connection with provided credentials"""
    print("🧪 API CONNECTION TEST")
    print("=" * 40)

    # Check for environment variables
    api_key = os.getenv('BINANCE_API_KEY')
    secret = os.getenv('BINANCE_SECRET')

    if not api_key or not secret:
        print("❌ No API credentials found!")
        print("\nTo test with real API credentials:")
        print("1. Copy .env.example to .env")
        print("2. Fill in your Binance API key and secret")
        print("3. Run: source .env && python test_api_connection.py")
        print("\nOR set environment variables:")
        print("export BINANCE_API_KEY='your_api_key'")
        print("export BINANCE_SECRET='your_secret'")
        return

    print(f"✅ Found API credentials")
    print(f"   API Key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else api_key}")
    print(f"   Secret:  {secret[:8]}...{secret[-4:] if len(secret) > 12 else secret}")

    # Initialize logger and registry
    logger = TradeLogger("api_test_" + str(int(time.time())))
    registry = await get_exchange_registry(logger)

    print("\n🔌 Testing Binance connection...")

    try:
        # Initialize exchange
        success = await registry.initialize_exchange('binance')

        if success:
            print("✅ Successfully connected to Binance!")

            # Test basic functionality
            print("\n📊 Testing basic functionality...")

            # Get supported symbols
            symbols = registry.get_supported_symbols('binance')
            print(f"   📈 Found {len(symbols)} trading pairs")

            # Show some popular symbols
            popular_symbols = [s for s in symbols if any(pair in s for pair in ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'])]
            print(f"   Popular pairs: {popular_symbols[:5]}")

            # Test capabilities
            capabilities = registry.get_exchange_capabilities('binance')
            print(f"   🛠️ Capabilities: {capabilities}")

            # Test unified ticker (if we have symbols)
            if symbols:
                test_symbol = 'BTC/USDT' if 'BTC/USDT' in symbols else symbols[0]
                print(f"\n💰 Testing ticker for {test_symbol}...")

                try:
                    ticker = await registry.get_unified_ticker(test_symbol, ['binance'])
                    if 'error' not in ticker:
                        print(f"   Price: ${ticker.get('best_bid', 'N/A')}")
                        print(f"   Spread: {ticker.get('spread_pct', 0):.4f}%")
                    else:
                        print(f"   ⚠️ Ticker error: {ticker['error']}")
                except Exception as e:
                    print(f"   ⚠️ Ticker test failed: {e}")

            print("\n🎉 API connection test completed successfully!")

        else:
            print("❌ Failed to connect to Binance")
            print("   Check your API credentials and permissions")
            print("   Make sure your API key has the required permissions")

    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        import traceback
        print("\n🔍 Full error details:")
        traceback.print_exc()
        print("\n   Common causes:")
        print("   - Invalid API credentials")
        print("   - Network connectivity issues")
        print("   - API rate limits")
        print("   - Insufficient API permissions")

    finally:
        # Clean up
        await registry.close_all()


if __name__ == "__main__":
    import time
    asyncio.run(test_api_connection())