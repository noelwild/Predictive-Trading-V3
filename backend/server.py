#!/usr/bin/env python3
"""
1-Second Trading System v3.0 - Development Server
AI-Powered Trading Platform with Auto-Browser Launch

Usage: python server.py
This will start the backend server and automatically open the frontend in your browser.
"""

import webbrowser
import time
import threading
import uvicorn
import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

def open_browser(url, delay=3):
    """Open browser after server starts"""
    def delayed_open():
        time.sleep(delay)
        print(f"\n🌐 Opening browser: {url}")
        print("🚀 1-Second Trading System v3.0 is now running!")
        print("📊 Dashboard: Real-time AI trading interface")
        print("🔄 Toggle: Switch between paper and live trading")
        print("🏦 Brokers: Configure your real trading APIs")
        print("\n⚠️  To stop the server, press Ctrl+C")
        print("=" * 60)
        
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"❌ Could not open browser automatically: {e}")
            print(f"📱 Manually open: {url}")
    
    thread = threading.Thread(target=delayed_open)
    thread.daemon = True
    thread.start()

def main():
    """Main server startup with browser launch"""
    
    # Server configuration
    HOST = "127.0.0.1"  # localhost for development
    PORT = 8001
    URL = f"http://{HOST}:{PORT}"
    
    print("🚀 Starting 1-Second Trading System v3.0...")
    print("=" * 60)
    print(f"🔧 Backend: FastAPI server on {URL}")
    print(f"🎨 Frontend: Vanilla HTML/CSS/JS interface")
    print(f"🧠 AI Engine: Claude-4-sonnet + ML models")
    print(f"💰 Trading: Paper + Live trading modes")
    print(f"🏦 Brokers: Alpaca, Interactive Brokers, Binance")
    print("=" * 60)
    
    # Import the trading system
    try:
        from trading_system import app
        print("✅ Trading system loaded successfully")
    except ImportError as e:
        print(f"❌ Error loading trading system: {e}")
        sys.exit(1)
    
    # Schedule browser opening
    open_browser(URL, delay=2)
    
    # Start the server
    try:
        uvicorn.run(
            app, 
            host=HOST, 
            port=PORT,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        print("👋 Thanks for using 1-Second Trading System v3.0!")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()