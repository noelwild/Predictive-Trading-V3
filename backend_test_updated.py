#!/usr/bin/env python3
"""
Updated Backend API Testing for 1-Second Trading System v3.0
Tests the actual implemented endpoints and WebSocket functionality
"""

import requests
import json
import sys
import time
import asyncio
import websockets
from datetime import datetime
from typing import Dict, Any, List

class TradingSystemTester:
    def __init__(self, base_url="https://speedtrader.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.ws_url = f"{base_url.replace('https://', 'wss://')}/ws"
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
        """Test root endpoint (serves HTML)"""
        print("\n🔍 Testing Root Endpoint (HTML)...")
        try:
            response = requests.get(self.base_url, timeout=30)
            if response.status_code == 200 and 'html' in response.headers.get('content-type', '').lower():
                if '1-Second Trading System' in response.text:
                    self.log_test("Root Endpoint", True, "HTML page served successfully")
                else:
                    self.log_test("Root Endpoint", False, "HTML doesn't contain expected title")
            else:
                self.log_test("Root Endpoint", False, f"Status {response.status_code}")
        except Exception as e:
            self.log_test("Root Endpoint", False, str(e))
    
    def test_static_files(self):
        """Test static file serving"""
        print("\n🔍 Testing Static Files...")
        
        # Test CSS
        try:
            response = requests.get(f"{self.base_url}/static/styles.css", timeout=30)
            if response.status_code == 200 and 'css' in response.headers.get('content-type', ''):
                self.log_test("CSS File", True, f"Size: {len(response.text)} bytes")
            else:
                self.log_test("CSS File", False, f"Status {response.status_code}")
        except Exception as e:
            self.log_test("CSS File", False, str(e))
        
        # Test JS
        try:
            response = requests.get(f"{self.base_url}/static/app.js", timeout=30)
            if response.status_code == 200 and 'javascript' in response.headers.get('content-type', ''):
                self.log_test("JS File", True, f"Size: {len(response.text)} bytes")
            else:
                self.log_test("JS File", False, f"Status {response.status_code}")
        except Exception as e:
            self.log_test("JS File", False, str(e))
    
    def test_api_status(self):
        """Test API status endpoint"""
        print("\n🔍 Testing API Status...")
        success, response = self.make_request('GET', '/status')
        
        if success and isinstance(response, dict):
            required_fields = ['system_state', 'positions', 'signals', 'health_check']
            missing_fields = [field for field in required_fields if field not in response]
            
            if not missing_fields:
                health = response.get('health_check', 'unknown')
                positions_count = len(response.get('positions', []))
                signals_count = len(response.get('signals', []))
                active_models = response.get('active_models', 0)
                
                self.log_test("API Status", True, 
                            f"Health: {health}, Positions: {positions_count}, Signals: {signals_count}, Models: {active_models}")
                return response
            else:
                self.log_test("API Status", False, f"Missing fields: {missing_fields}")
        else:
            self.log_test("API Status", False, str(response))
        return None
    
    def test_brokers_endpoint(self):
        """Test brokers endpoint"""
        print("\n🔍 Testing Brokers Endpoint...")
        success, response = self.make_request('GET', '/brokers')
        
        if success and isinstance(response, list):
            broker_count = len(response)
            broker_types = [b.get('broker_type', 'unknown') for b in response]
            self.log_test("GET Brokers", True, f"Found {broker_count} brokers: {broker_types}")
            
            # Check broker structure
            if response:
                broker = response[0]
                required_fields = ['name', 'broker_type', 'sandbox']
                missing_fields = [field for field in required_fields if field not in broker]
                
                if not missing_fields:
                    self.log_test("Broker Structure", True, "All required fields present")
                else:
                    self.log_test("Broker Structure", False, f"Missing fields: {missing_fields}")
            
            return response
        else:
            self.log_test("GET Brokers", False, str(response))
        return []
    
    def test_performance_endpoint(self):
        """Test performance endpoint"""
        print("\n🔍 Testing Performance Endpoint...")
        success, response = self.make_request('GET', '/performance')
        
        if success and isinstance(response, dict):
            summary = response.get('summary', {})
            daily_metrics = response.get('daily_metrics', [])
            
            if 'total_pnl' in summary and 'avg_win_rate' in summary:
                total_pnl = summary['total_pnl']
                win_rate = summary['avg_win_rate']
                days = len(daily_metrics)
                
                self.log_test("Performance Metrics", True, 
                            f"P&L: ${total_pnl:.2f}, Win Rate: {win_rate:.1%}, Days: {days}")
            else:
                self.log_test("Performance Metrics", False, "Invalid summary structure")
        else:
            self.log_test("Performance Metrics", False, str(response))
    
    async def test_websocket_connection(self):
        """Test WebSocket connection and communication"""
        print("\n🔍 Testing WebSocket Connection...")
        
        try:
            # Test WebSocket connection
            async with websockets.connect(self.ws_url, timeout=10) as websocket:
                self.log_test("WebSocket Connection", True, "Connected successfully")
                
                # Test sending a message
                test_message = {
                    "type": "test_connection",
                    "timestamp": datetime.now().isoformat()
                }
                
                await websocket.send(json.dumps(test_message))
                self.log_test("WebSocket Send", True, "Message sent successfully")
                
                # Try to receive a response (with timeout)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    response_data = json.loads(response)
                    self.log_test("WebSocket Receive", True, f"Received: {response_data.get('type', 'unknown')}")
                except asyncio.TimeoutError:
                    self.log_test("WebSocket Receive", False, "No response received within 5 seconds")
                except json.JSONDecodeError:
                    self.log_test("WebSocket Receive", False, "Invalid JSON response")
                
        except websockets.exceptions.ConnectionClosed:
            self.log_test("WebSocket Connection", False, "Connection closed unexpectedly")
        except websockets.exceptions.InvalidURI:
            self.log_test("WebSocket Connection", False, "Invalid WebSocket URI")
        except Exception as e:
            self.log_test("WebSocket Connection", False, f"Connection error: {str(e)}")
    
    def test_system_architecture(self):
        """Test system architecture validation"""
        print("\n🔍 Testing System Architecture...")
        
        # Test that it's serving vanilla HTML/CSS/JS (not React)
        try:
            response = requests.get(self.base_url, timeout=30)
            html_content = response.text
            
            # Check for React indicators (should NOT be present)
            react_indicators = ['react', 'ReactDOM', 'jsx', 'babel']
            react_found = any(indicator in html_content.lower() for indicator in react_indicators)
            
            if not react_found:
                self.log_test("Vanilla Frontend", True, "No React dependencies found")
            else:
                self.log_test("Vanilla Frontend", False, "React indicators found in HTML")
            
            # Check for vanilla JS indicators
            vanilla_indicators = ['TradingSystemApp', 'WebSocket', 'addEventListener']
            vanilla_found = any(indicator in html_content for indicator in vanilla_indicators)
            
            if vanilla_found:
                self.log_test("Vanilla JavaScript", True, "Vanilla JS patterns detected")
            else:
                self.log_test("Vanilla JavaScript", False, "Vanilla JS patterns not found")
                
        except Exception as e:
            self.log_test("Architecture Test", False, str(e))
    
    def test_live_trading_features(self):
        """Test live trading feature indicators"""
        print("\n🔍 Testing Live Trading Features...")
        
        # Check if brokers are configured for live trading
        success, brokers = self.make_request('GET', '/brokers')
        
        if success and isinstance(brokers, list):
            live_brokers = [b for b in brokers if not b.get('sandbox', True)]
            paper_brokers = [b for b in brokers if b.get('sandbox', True)]
            
            self.log_test("Broker Configuration", True, 
                        f"Paper: {len(paper_brokers)}, Live: {len(live_brokers)}")
            
            # Check for broker types that support live trading
            supported_types = ['alpaca', 'interactive_brokers', 'binance']
            configured_types = [b.get('broker_type') for b in brokers]
            live_capable = any(bt in supported_types for bt in configured_types)
            
            if live_capable:
                self.log_test("Live Trading Capability", True, f"Supported brokers: {configured_types}")
            else:
                self.log_test("Live Trading Capability", False, "No live trading brokers configured")
        else:
            self.log_test("Broker Configuration", False, "Could not retrieve broker data")
    
    def run_comprehensive_test(self):
        """Run all tests in sequence"""
        print("🚀 Starting Updated Backend Testing for 1-Second Trading System v3.0")
        print("=" * 80)
        
        start_time = time.time()
        
        # Test architecture and basic functionality
        self.test_system_architecture()
        self.test_root_endpoint()
        self.test_static_files()
        
        # Test API endpoints
        self.test_api_status()
        self.test_brokers_endpoint()
        self.test_performance_endpoint()
        
        # Test live trading features
        self.test_live_trading_features()
        
        # Test WebSocket (async)
        try:
            asyncio.run(self.test_websocket_connection())
        except Exception as e:
            self.log_test("WebSocket Test", False, f"Async test failed: {str(e)}")
        
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
            print("✅ System is functioning well - architecture validated, APIs working")
        elif self.tests_passed >= self.tests_run * 0.6:
            print("⚠️  System has some issues - partial functionality working")
        else:
            print("❌ System has major issues - significant functionality broken")
        
        return self.tests_passed >= self.tests_run * 0.7

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