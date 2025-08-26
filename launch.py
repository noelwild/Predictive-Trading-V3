#!/usr/bin/env python3
"""
🚀 1-Second Trading System v3.0 - Quick Launcher
One-command startup with automatic browser launch

Run this file from anywhere to start your AI trading system!
"""

import os
import sys
import subprocess
import webbrowser
import time
import threading
from pathlib import Path

def print_banner():
    """Print startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║         🚀 1-Second Trading System v3.0                      ║
    ║                                                              ║
    ║    🧠 AI-Powered High-Frequency Trading Platform             ║
    ║    📊 Claude-4-sonnet + Multi-Timeframe ML Models           ║
    ║    💰 Paper Trading + Live Trading with Real Brokers        ║
    ║    🎨 Professional Vanilla HTML/CSS/JS Interface             ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_requirements():
    """Check if system requirements are met"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check if we're in the right directory
    backend_dir = Path(__file__).parent / "backend"
    if not backend_dir.exists():
        print("❌ Backend directory not found")
        print("💡 Make sure you're running this from the project root directory")
        return False
    
    print("✅ Project structure found")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Checking dependencies...")
    
    backend_dir = Path(__file__).parent / "backend"
    requirements_file = backend_dir / "requirements.txt"
    
    if requirements_file.exists():
        print("📥 Installing Python dependencies...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=True, capture_output=True)
            print("✅ Dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Some dependencies may have failed to install: {e}")
            print("🚀 Continuing anyway - core functionality should work")
    
    return True

def start_system():
    """Start the trading system"""
    print("🚀 Starting AI Trading System...")
    
    backend_dir = Path(__file__).parent / "backend"
    server_file = backend_dir / "server.py"
    
    if not server_file.exists():
        print("❌ Server file not found")
        return False
    
    # Change to backend directory
    os.chdir(backend_dir)
    
    # Start the server
    try:
        subprocess.run([sys.executable, "server.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
        return False
    
    return True

def main():
    """Main launcher function"""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    print("\n🎯 System Ready! Starting server and opening browser...")
    print("📱 URL: http://127.0.0.1:8001")
    print("🔄 Mode: Paper trading (safe) - switch to live for real money")
    print("🏦 Brokers: Add your API credentials for live trading")
    print("\n" + "=" * 60)
    
    # Start the system
    if not start_system():
        input("\nPress Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Thanks for using 1-Second Trading System v3.0!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        input("Press Enter to exit...")
        sys.exit(1)