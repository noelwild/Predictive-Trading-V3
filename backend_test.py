#!/usr/bin/env python3
"""
Comprehensive Backend API Testing for 1-Second Trading System v3.0
Tests all major API endpoints and functionality
"""

import requests
import json
import sys
import time
from datetime import datetime
from typing import Dict, Any, List

class TradingSystemTester:
    def __init__(self, base_url="https://speedtrader.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.failed_tests = []
        
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test results"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {test_name}: PASSED {details}")
        else:
            self.failed_tests.append(f"{test_name}: {details}")
            print(f"❌ {test_name}: FAILED {details}")
    
    def make_request(self, method: str, endpoint: str, data: Dict = None, expected_status: int = 200) -> tuple:
        """Make HTTP request and return success status and response"""
        url = f"{self.api_url}/{endpoint.lstrip('/')}"
        headers = {'Content-Type': 'application/json'}
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method.upper() == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=30)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=30)
            else:
                return False, f"Unsupported method: {method}"
            
            success = response.status_code == expected_status
            
            if success:
                try:
                    response_data = response.json()
                    return True, response_data
                except:
                    return True, response.text
            else:
                return False, f"Status {response.status_code}: {response.text[:200]}"
                
        except requests.exceptions.Timeout:
            return False, "Request timeout (30s)"
        except requests.exceptions.ConnectionError:
            return False, "Connection error - backend may be down"
        except Exception as e:
            return False, f"Request error: {str(e)}"
    
    def test_root_endpoint(self):
        """Test root API endpoint"""
        print("\n🔍 Testing Root Endpoint...")
        success, response = self.make_request('GET', '/')
        
        if success and isinstance(response, dict):
            if 'message' in response and '1-Second Trading System' in response['message']:
                self.log_test("Root Endpoint", True, f"Message: {response.get('message', '')}")
            else:
                self.log_test("Root Endpoint", False, "Invalid response format")
        else:
            self.log_test("Root Endpoint", False, str(response))
    
    def test_system_status(self):
        """Test system status endpoint"""
        print("\n🔍 Testing System Status...")
        success, response = self.make_request('GET', '/system/status')
        
        if success and isinstance(response, dict):
            required_fields = ['system_state', 'health_check', 'uptime']
            missing_fields = [field for field in required_fields if field not in response]
            
            if not missing_fields:
                health_status = response.get('system_state', {}).get('health_status', 'unknown')
                phase = response.get('system_state', {}).get('phase', 'unknown')
                self.log_test("System Status", True, f"Health: {health_status}, Phase: {phase}")
                return response
            else:
                self.log_test("System Status", False, f"Missing fields: {missing_fields}")
        else:
            self.log_test("System Status", False, str(response))
        return None
    
    def test_broker_endpoints(self):
        """Test broker configuration endpoints"""
        print("\n🔍 Testing Broker Endpoints...")
        
        # Test GET brokers
        success, response = self.make_request('GET', '/brokers')
        if success:
            self.log_test("GET Brokers", True, f"Found {len(response) if isinstance(response, list) else 0} brokers")
        else:
            self.log_test("GET Brokers", False, str(response))
        
        # Test POST broker
        test_broker = {
            "name": "Test Broker",
            "broker_type": "alpaca",
            "api_key": "test_key_123",
            "api_secret": "test_secret_456",
            "sandbox": True
        }
        
        success, response = self.make_request('POST', '/brokers', test_broker, 200)
        if success:
            self.log_test("POST Broker", True, "Broker configuration added")
        else:
            self.log_test("POST Broker", False, str(response))
    
    def test_symbols_endpoints(self):
        """Test trading symbols endpoints"""
        print("\n🔍 Testing Symbols Endpoints...")
        
        # Test GET symbols
        success, response = self.make_request('GET', '/symbols')
        if success:
            self.log_test("GET Symbols", True, f"Found {len(response) if isinstance(response, list) else 0} symbols")
        else:
            self.log_test("GET Symbols", False, str(response))
        
        # Test POST symbol
        test_symbol = {
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            "asset_type": "stock",
            "is_active": True
        }
        
        success, response = self.make_request('POST', '/symbols', test_symbol, 200)
        if success:
            self.log_test("POST Symbol", True, "Trading symbol added")
        else:
            self.log_test("POST Symbol", False, str(response))
    
    def test_data_collection(self):
        """Test data collection workflow"""
        print("\n🔍 Testing Data Collection...")
        
        test_symbols = ["AAPL", "MSFT", "GOOGL"]
        success, response = self.make_request('POST', '/system/start-data-collection', 
                                            {"symbols": test_symbols}, 200)
        
        if success and isinstance(response, dict):
            if response.get('status') == 'started':
                self.log_test("Start Data Collection", True, f"Symbols: {test_symbols}")
                time.sleep(2)  # Give it time to start
            else:
                self.log_test("Start Data Collection", False, f"Unexpected status: {response.get('status')}")
        else:
            self.log_test("Start Data Collection", False, str(response))
    
    def test_model_training(self):
        """Test ML model training"""
        print("\n🔍 Testing Model Training...")
        
        success, response = self.make_request('POST', '/system/train-models', {}, 200)
        
        if success and isinstance(response, dict):
            if 'training_started' in response.get('status', ''):
                models = response.get('models', [])
                self.log_test("Train Models", True, f"Training models: {models}")
                time.sleep(3)  # Give training time to start
            else:
                self.log_test("Train Models", False, f"Unexpected response: {response}")
        else:
            self.log_test("Train Models", False, str(response))
    
    def test_predictions(self):
        """Test prediction endpoints"""
        print("\n🔍 Testing Predictions...")
        
        test_symbols = ["AAPL", "MSFT", "GOOGL"]
        
        for symbol in test_symbols:
            success, response = self.make_request('GET', f'/predictions/{symbol}')
            
            if success and isinstance(response, dict):
                predictions = response.get('predictions', {})
                if predictions:
                    timeframes = list(predictions.keys())
                    self.log_test(f"Predictions {symbol}", True, f"Timeframes: {timeframes}")
                    
                    # Check if Claude analysis is present
                    for tf, pred in predictions.items():
                        if isinstance(pred, dict) and 'claude_analysis' in pred:
                            claude_data = pred['claude_analysis']
                            if isinstance(claude_data, dict) and 'market_regime' in claude_data:
                                self.log_test(f"Claude Analysis {symbol}-{tf}", True, 
                                            f"Regime: {claude_data.get('market_regime')}")
                                break
                else:
                    self.log_test(f"Predictions {symbol}", False, "No predictions returned")
            else:
                self.log_test(f"Predictions {symbol}", False, str(response))
    
    def test_trading_control(self):
        """Test trading start/stop functionality"""
        print("\n🔍 Testing Trading Control...")
        
        # Test start trading
        success, response = self.make_request('POST', '/system/start-trading', {}, 200)
        if success and isinstance(response, dict):
            if 'trading_started' in response.get('status', ''):
                self.log_test("Start Trading", True, f"Mode: {response.get('mode', 'unknown')}")
                time.sleep(2)
            else:
                self.log_test("Start Trading", False, f"Unexpected response: {response}")
        else:
            self.log_test("Start Trading", False, str(response))
        
        # Test stop trading
        success, response = self.make_request('POST', '/system/stop-trading', {}, 200)
        if success and isinstance(response, dict):
            if 'trading_stopped' in response.get('status', ''):
                self.log_test("Stop Trading", True, "Trading stopped successfully")
            else:
                self.log_test("Stop Trading", False, f"Unexpected response: {response}")
        else:
            self.log_test("Stop Trading", False, str(response))
    
    def test_data_endpoints(self):
        """Test data retrieval endpoints"""
        print("\n🔍 Testing Data Endpoints...")
        
        # Test positions
        success, response = self.make_request('GET', '/positions')
        if success:
            positions_count = len(response) if isinstance(response, list) else 0
            self.log_test("GET Positions", True, f"Found {positions_count} positions")
        else:
            self.log_test("GET Positions", False, str(response))
        
        # Test signals
        success, response = self.make_request('GET', '/signals')
        if success:
            signals_count = len(response) if isinstance(response, list) else 0
            self.log_test("GET Signals", True, f"Found {signals_count} signals")
        else:
            self.log_test("GET Signals", False, str(response))
        
        # Test performance metrics
        success, response = self.make_request('GET', '/performance/metrics')
        if success and isinstance(response, dict):
            summary = response.get('summary', {})
            if summary:
                total_pnl = summary.get('total_pnl', 0)
                win_rate = summary.get('avg_win_rate', 0)
                self.log_test("Performance Metrics", True, f"P&L: ${total_pnl}, Win Rate: {win_rate:.1%}")
            else:
                self.log_test("Performance Metrics", True, "No performance data yet")
        else:
            self.log_test("Performance Metrics", False, str(response))
        
        # Test market data
        success, response = self.make_request('GET', '/market-data/AAPL?limit=10')
        if success:
            data_count = len(response) if isinstance(response, list) else 0
            self.log_test("Market Data", True, f"Found {data_count} data points for AAPL")
        else:
            self.log_test("Market Data", False, str(response))
    
    def test_system_config(self):
        """Test system configuration endpoints"""
        print("\n🔍 Testing System Configuration...")
        
        # Test GET config
        success, response = self.make_request('GET', '/system/config')
        if success and isinstance(response, dict):
            max_position = response.get('max_position_size', 0)
            risk_per_trade = response.get('risk_per_trade', 0)
            self.log_test("GET System Config", True, f"Max Position: ${max_position}, Risk: {risk_per_trade}")
        else:
            self.log_test("GET System Config", False, str(response))
    
    def run_comprehensive_test(self):
        """Run all tests in sequence"""
        print("🚀 Starting Comprehensive Backend API Testing for 1-Second Trading System v3.0")
        print("=" * 80)
        
        start_time = time.time()
        
        # Run all test categories
        self.test_root_endpoint()
        self.test_system_status()
        self.test_broker_endpoints()
        self.test_symbols_endpoints()
        self.test_system_config()
        self.test_data_collection()
        
        # Wait a bit for data collection to process
        print("\n⏳ Waiting for data collection to process...")
        time.sleep(5)
        
        self.test_model_training()
        
        # Wait for model training to start
        print("\n⏳ Waiting for model training to initialize...")
        time.sleep(8)
        
        self.test_predictions()
        self.test_trading_control()
        self.test_data_endpoints()
        
        # Print final results
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "=" * 80)
        print("📊 FINAL TEST RESULTS")
        print("=" * 80)
        print(f"Total Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {len(self.failed_tests)}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        print(f"Test Duration: {duration:.1f} seconds")
        
        if self.failed_tests:
            print("\n❌ FAILED TESTS:")
            for i, failure in enumerate(self.failed_tests, 1):
                print(f"{i}. {failure}")
        
        print("\n🎯 KEY FINDINGS:")
        if self.tests_passed >= self.tests_run * 0.8:
            print("✅ Backend API is functioning well - most endpoints working correctly")
        elif self.tests_passed >= self.tests_run * 0.6:
            print("⚠️  Backend API has some issues - partial functionality working")
        else:
            print("❌ Backend API has major issues - significant functionality broken")
        
        return self.tests_passed == self.tests_run

def main():
    """Main test execution"""
    tester = TradingSystemTester()
    
    try:
        success = tester.run_comprehensive_test()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n⚠️ Testing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n💥 Testing failed with error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())