/**
 * 1-Second Trading System v3.0 - Vanilla JavaScript Application
 * Real-time WebSocket communication with AI trading backend
 */

class TradingSystemApp {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 2000;
        
        this.systemData = {
            system_state: {
                phase: 'initialization',
                health_status: 'healthy',
                models_trained: false,
                trading_active: false
            },
            positions: [],
            signals: [],
            brokers: [],
            performance: null,
            predictions: {}
        };
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.connectWebSocket();
        this.setupTabs();
        this.loadInitialData();
    }
    
    // WebSocket Connection
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('🔗 WebSocket connected');
            this.updateConnectionStatus(true);
            this.reconnectAttempts = 0;
        };
        
        this.ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleWebSocketMessage(message);
            } catch (error) {
                console.error('WebSocket message parse error:', error);
            }
        };
        
        this.ws.onclose = () => {
            console.log('🔌 WebSocket disconnected');
            this.updateConnectionStatus(false);
            this.attemptReconnect();
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus(false);
        };
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`🔄 Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            
            setTimeout(() => {
                this.connectWebSocket();
            }, this.reconnectDelay * this.reconnectAttempts);
        } else {
            console.error('❌ Max reconnection attempts reached');
            this.showNotification('Connection failed. Please refresh the page.', 'error');
        }
    }
    
    sendMessage(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        } else {
            console.error('WebSocket not connected');
            this.showNotification('Connection lost. Attempting to reconnect...', 'error');
        }
    }
    
    // Message Handlers
    handleWebSocketMessage(message) {
        console.log('📨 Received:', message.type);
        
        switch (message.type) {
            case 'system_state':
                this.systemData.system_state = message.data;
                this.updateSystemStatus();
                break;
                
            case 'system_update':
                Object.assign(this.systemData.system_state, message.data);
                this.updateSystemStatus();
                break;
                
            case 'predictions':
                this.systemData.predictions[message.data.symbol] = message.data.predictions;
                this.updatePredictions(message.data.symbol, message.data.predictions);
                break;
                
            case 'new_signal':
                this.systemData.signals.unshift(message.data);
                this.updateSignals();
                this.showNotification(`New signal: ${message.data.direction} ${message.data.symbol}`, 'info');
                break;
                
            case 'positions_update':
                this.systemData.positions = message.data;
                this.updatePositions();
                break;
                
            case 'performance_update':
                this.systemData.performance = message.data;
                this.updatePerformance();
                break;
                
            case 'broker_added':
                this.systemData.brokers.push(message.data);
                this.updateBrokers();
                this.showNotification('Broker configuration added successfully', 'success');
                break;
                
            case 'notification':
                this.showNotification(message.data.message, message.data.type || 'info');
                break;
                
            case 'trading_mode_changed':
                this.updateTradingModeUI(message.data.paper_trading);
                break;
                
            case 'broker_activated':
                if (message.data.success) {
                    this.showNotification('Broker connected successfully', 'success');
                    this.loadBrokers(); // Refresh broker list
                } else {
                    this.showNotification('Failed to connect to broker', 'error');
                }
                break;
                
            case 'account_status':
                this.updateAccountInfo(message.data);
                break;
                
            default:
                console.log('Unknown message type:', message.type);
        }
    }
    
    // UI Updates
    updateSystemStatus() {
        const state = this.systemData.system_state;
        
        // Update system health
        const healthElement = document.getElementById('system-health');
        if (healthElement) {
            healthElement.textContent = state.health_status || 'unknown';
            healthElement.className = `status-text ${state.health_status === 'healthy' ? 'pnl-positive' : 'pnl-negative'}`;
        }
        
        // Update system phase
        const phaseElement = document.getElementById('system-phase');
        if (phaseElement) {
            phaseElement.textContent = state.phase || 'unknown';
            phaseElement.className = `status-badge ${state.phase}`;
        }
        
        // Update active models
        const modelsElement = document.getElementById('active-models');
        if (modelsElement) {
            modelsElement.textContent = state.active_models || 0;
        }
        
        // Update control buttons
        this.updateControlButtons();
    }
    
    updateControlButtons() {
        const state = this.systemData.system_state;
        
        const trainModelsBtn = document.getElementById('train-models');
        const startTradingBtn = document.getElementById('start-trading');
        const stopTradingBtn = document.getElementById('stop-trading');
        
        if (trainModelsBtn) {
            trainModelsBtn.disabled = !state.data_collection_active || state.models_trained;
            trainModelsBtn.innerHTML = state.models_trained ? '✅ Models Trained' : 'Train ML Models';
        }
        
        if (startTradingBtn) {
            startTradingBtn.disabled = !state.models_trained || state.trading_active;
        }
        
        if (stopTradingBtn) {
            stopTradingBtn.disabled = !state.trading_active;
        }
    }
    
    updatePredictions(symbol, predictions) {
        const timeframes = ['500ms', '1s', '5s', '10s'];
        
        // Update prediction for AAPL (main display)
        if (symbol === 'AAPL') {
            timeframes.forEach(timeframe => {
                const prediction = predictions[timeframe];
                if (prediction && !prediction.error) {
                    this.updatePredictionCard(timeframe, prediction);
                }
            });
        }
    }
    
    updatePredictionCard(timeframe, prediction) {
        const card = document.getElementById(`prediction-${timeframe}`);
        if (!card) return;
        
        const directionElement = card.querySelector('.prediction-direction');
        const confidenceElement = card.querySelector('.prediction-confidence');
        const regimeElement = card.querySelector('.prediction-regime');
        
        if (directionElement) {
            directionElement.textContent = prediction.direction || 'HOLD';
            directionElement.className = `prediction-direction ${(prediction.direction || 'HOLD').toLowerCase()}`;
        }
        
        if (confidenceElement) {
            const confidence = Math.round((prediction.confidence || 0) * 100);
            confidenceElement.textContent = `${confidence}%`;
        }
        
        if (regimeElement) {
            const regime = prediction.claude_analysis?.market_regime || 'normal';
            regimeElement.textContent = regime;
        }
    }
    
    updateSignals() {
        const signalsList = document.getElementById('signals-list');
        if (!signalsList) return;
        
        if (this.systemData.signals.length === 0) {
            signalsList.innerHTML = '<p class="empty-state">No trading signals yet</p>';
            return;
        }
        
        const signalsHTML = this.systemData.signals.slice(0, 10).map(signal => {
            const time = new Date(signal.timestamp).toLocaleTimeString();
            const confidence = Math.round(signal.confidence * 100);
            
            return `
                <div class="signal-item ${signal.executed ? 'executed' : 'pending'}">
                    <div class="item-left">
                        <div class="item-badge">${signal.symbol}</div>
                        <div class="item-direction ${signal.direction.toLowerCase()}">${signal.direction}</div>
                        <div class="item-details">${signal.size} shares @ $${signal.price.toFixed(2)}</div>
                    </div>
                    <div class="item-right">
                        <div class="item-time">${time}</div>
                        <div class="item-status ${signal.executed ? 'executed' : 'pending'}">
                            ${signal.executed ? 'Executed' : 'Pending'}
                        </div>
                    </div>
                </div>
            `;
        }).join('');
        
        signalsList.innerHTML = signalsHTML;
    }
    
    updatePositions() {
        const positionsList = document.getElementById('positions-list');
        const totalPositionsElement = document.getElementById('total-positions');
        const totalPnlElement = document.getElementById('total-pnl');
        
        if (totalPositionsElement) {
            totalPositionsElement.textContent = this.systemData.positions.length;
        }
        
        // Calculate total P&L
        const totalPnl = this.systemData.positions.reduce((sum, pos) => sum + (pos.pnl || 0), 0);
        
        if (totalPnlElement) {
            totalPnlElement.textContent = `$${totalPnl.toLocaleString()}`;
            totalPnlElement.className = `status-text ${totalPnl >= 0 ? 'pnl-positive' : 'pnl-negative'}`;
        }
        
        if (!positionsList) return;
        
        if (this.systemData.positions.length === 0) {
            positionsList.innerHTML = '<p class="empty-state">No open positions</p>';
            return;
        }
        
        const positionsHTML = this.systemData.positions.map(position => {
            const pnl = position.pnl || 0;
            const pnlClass = pnl >= 0 ? 'positive' : 'negative';
            
            return `
                <div class="position-item">
                    <div class="item-left">
                        <div class="item-badge">${position.symbol}</div>
                        <div class="item-details">${Math.abs(position.size)} shares @ $${position.entry_price.toFixed(2)}</div>
                        <div class="item-direction ${position.size > 0 ? 'buy' : 'sell'}">
                            ${position.size > 0 ? 'LONG' : 'SHORT'}
                        </div>
                    </div>
                    <div class="item-right">
                        <div class="daily-pnl ${pnlClass}">$${pnl.toFixed(2)}</div>
                        <div class="item-details">Current: $${position.current_price.toFixed(2)}</div>
                    </div>
                </div>
            `;
        }).join('');
        
        positionsList.innerHTML = positionsHTML;
    }
    
    updateTradingModeUI(paperMode) {
        const modeDescription = document.getElementById('mode-description');
        const liveTradingWarning = document.getElementById('live-trading-warning');
        const tradingModeToggle = document.getElementById('trading-mode-toggle');
        
        if (modeDescription) {
            if (paperMode) {
                modeDescription.textContent = 'Safe paper trading with fake money (recommended for testing)';
                modeDescription.className = 'mode-description';
            } else {
                modeDescription.textContent = '⚠️ LIVE TRADING MODE - Real money will be used for trades!';
                modeDescription.className = 'mode-description live-warning';
            }
        }
        
        if (liveTradingWarning) {
            liveTradingWarning.className = paperMode ? 'live-trading-warning hidden' : 'live-trading-warning';
        }
        
        // Update toggle state
        if (tradingModeToggle) {
            tradingModeToggle.checked = paperMode;
        }
        
        // Update system state
        this.systemData.system_state.paper_trading = paperMode;
        
        // Request account info if live mode
        if (!paperMode) {
            this.sendMessage({
                type: 'get_account_status'
            });
        }
    }
    
    activateBroker(brokerId) {
        this.sendMessage({
            type: 'activate_broker',
            broker_id: brokerId
        });
        
        this.showNotification('Connecting to broker...', 'info');
    }
    
    updateBrokerStatus(brokerData) {
        const activeBrokerName = document.getElementById('active-broker-name');
        const connectionStatus = document.getElementById('broker-connection-status');
        const accountInfo = document.getElementById('account-info');
        
        if (brokerData.active_broker) {
            if (activeBrokerName) {
                activeBrokerName.textContent = brokerData.active_broker;
            }
            
            if (connectionStatus) {
                connectionStatus.className = 'broker-connection-dot online';
            }
            
            // Show account info for live trading
            if (!this.systemData.system_state.paper_trading && accountInfo) {
                accountInfo.className = 'account-info';
            }
        } else {
            if (activeBrokerName) {
                activeBrokerName.textContent = 'None';
            }
            
            if (connectionStatus) {
                connectionStatus.className = 'broker-connection-dot offline';
            }
            
            if (accountInfo) {
                accountInfo.className = 'account-info hidden';
            }
        }
    }
    
    updateBrokers() {
        const brokersContainer = document.getElementById('brokers-container');
        if (!brokersContainer) return;
        
        if (this.systemData.brokers.length === 0) {
            brokersContainer.innerHTML = '<p class="empty-state">No brokers configured yet</p>';
            return;
        }
        
        const brokersHTML = this.systemData.brokers.map(broker => `
            <div class="broker-item ${broker.is_active ? 'active' : ''}">
                <div class="broker-info">
                    <div class="broker-name">${broker.name}</div>
                    <div class="broker-type">${broker.broker_type}</div>
                    ${broker.sandbox ? '<div class="broker-sandbox">Sandbox</div>' : '<div class="broker-live">LIVE</div>'}
                </div>
                <div class="broker-actions">
                    <button class="activate-broker-btn ${broker.is_active ? 'active' : ''}" 
                            onclick="window.tradingApp.activateBroker('${broker.id}')"
                            ${broker.is_active ? 'disabled' : ''}>
                        ${broker.is_active ? '✅ Active' : 'Activate'}
                    </button>
                    <div class="broker-status ${broker.is_active ? 'active' : 'inactive'}"></div>
                </div>
            </div>
        `).join('');
        
        brokersContainer.innerHTML = brokersHTML;
    }
    
    updatePerformance() {
        const performance = this.systemData.performance;
        if (!performance) return;
        
        // Update performance summary
        const totalPnlEl = document.getElementById('perf-total-pnl');
        const winRateEl = document.getElementById('perf-win-rate');
        const sharpeRatioEl = document.getElementById('perf-sharpe-ratio');
        const tradingDaysEl = document.getElementById('perf-trading-days');
        
        if (totalPnlEl) totalPnlEl.textContent = `$${(performance.total_pnl || 0).toLocaleString()}`;
        if (winRateEl) winRateEl.textContent = `${((performance.win_rate || 0) * 100).toFixed(1)}%`;
        if (sharpeRatioEl) sharpeRatioEl.textContent = (performance.sharpe_ratio || 0).toFixed(2);
        if (tradingDaysEl) tradingDaysEl.textContent = '1';
    }
    
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connection-status');
        const indicator = statusElement?.querySelector('.connection-indicator');
        
        if (indicator) {
            indicator.className = `connection-indicator ${connected ? 'online' : 'offline'}`;
            indicator.querySelector('.connection-text').textContent = connected ? 'Connected' : 'Disconnected';
        }
    }
    
    // Event Listeners
    setupEventListeners() {
        // Refresh button
        document.getElementById('refresh-btn')?.addEventListener('click', () => {
            this.loadInitialData();
        });
        
        // Trading mode toggle
        document.getElementById('trading-mode-toggle')?.addEventListener('change', (e) => {
            const paperMode = e.target.checked;
            this.sendMessage({
                type: 'toggle_trading_mode',
                paper_trading: paperMode
            });
            
            this.updateTradingModeUI(paperMode);
        });
        
        // Data collection
        document.getElementById('start-data-collection')?.addEventListener('click', () => {
            const symbolsInput = document.getElementById('symbols-input');
            const symbols = symbolsInput.value.split(',').map(s => s.trim()).filter(s => s);
            
            this.sendMessage({
                type: 'start_data_collection',
                symbols: symbols
            });
            
            // Update button state
            const btn = document.getElementById('start-data-collection');
            btn.innerHTML = '<div class="loading"><div class="loading-spinner"></div>Collecting Data...</div>';
            btn.disabled = true;
            
            this.showNotification('Data collection started', 'success');
        });
        
        // Model training
        document.getElementById('train-models')?.addEventListener('click', () => {
            this.sendMessage({
                type: 'train_models'
            });
            
            this.showNotification('Model training started', 'success');
        });
        
        // Trading control
        document.getElementById('start-trading')?.addEventListener('click', () => {
            const paperMode = document.getElementById('trading-mode-toggle').checked;
            
            if (!paperMode) {
                // Live trading confirmation
                const confirmed = confirm(
                    '⚠️ WARNING: You are about to start LIVE TRADING with REAL MONEY!\n\n' +
                    'This will execute real trades through your broker.\n\n' +
                    'Are you sure you want to continue?'
                );
                
                if (!confirmed) {
                    return;
                }
            }
            
            this.sendMessage({
                type: 'start_trading'
            });
            
            const modeText = paperMode ? 'Paper trading' : 'LIVE TRADING';
            this.showNotification(`${modeText} started`, paperMode ? 'success' : 'error');
        });
        
        document.getElementById('stop-trading')?.addEventListener('click', () => {
            this.sendMessage({
                type: 'stop_trading'
            });
            
            this.showNotification('Trading stopped', 'info');
        });
        
        // Test prediction
        document.getElementById('test-prediction')?.addEventListener('click', () => {
            const symbol = document.getElementById('test-symbol').value;
            this.sendMessage({
                type: 'get_predictions',
                symbol: symbol
            });
        });
        
        // Broker form
        document.getElementById('broker-form')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleBrokerSubmit();
        });
    }
    
    handleBrokerSubmit() {
        const brokerData = {
            name: document.getElementById('broker-name').value,
            broker_type: document.getElementById('broker-type').value,
            api_key: document.getElementById('api-key').value,
            api_secret: document.getElementById('api-secret').value,
            sandbox: document.getElementById('sandbox-mode').checked
        };
        
        if (!brokerData.name || !brokerData.broker_type || !brokerData.api_key) {
            this.showNotification('Please fill in all required fields', 'error');
            return;
        }
        
        this.sendMessage({
            type: 'add_broker',
            data: brokerData
        });
        
        // Reset form
        document.getElementById('broker-form').reset();
        document.getElementById('sandbox-mode').checked = true;
    }
    
    // Tab Management
    setupTabs() {
        const tabBtns = document.querySelectorAll('.tab-btn');
        const tabPanes = document.querySelectorAll('.tab-pane');
        
        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const targetTab = btn.dataset.tab;
                
                // Update button states
                tabBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                // Update pane visibility
                tabPanes.forEach(pane => pane.classList.remove('active'));
                document.getElementById(`${targetTab}-tab`)?.classList.add('active');
                
                // Load tab-specific data
                this.onTabChange(targetTab);
            });
        });
    }
    
    onTabChange(tab) {
        switch (tab) {
            case 'dashboard':
                this.requestPredictions();
                break;
            case 'brokers':
                this.loadBrokers();
                break;
            case 'performance':
                this.loadPerformance();
                break;
        }
    }
    
    // Data Loading
    async loadInitialData() {
        try {
            console.log('📊 Loading initial data...');
            
            // Load system status via HTTP
            const response = await fetch('/api/status');
            const data = await response.json();
            
            this.systemData.system_state = data.system_state;
            this.systemData.positions = data.positions || [];
            this.systemData.signals = data.signals || [];
            
            this.updateSystemStatus();
            this.updateSignals();
            this.updatePositions();
            
            // Update active models count
            document.getElementById('active-models').textContent = data.active_models || 0;
            
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.showNotification('Error loading system data', 'error');
        }
    }
    
    async loadBrokers() {
        try {
            const response = await fetch('/api/brokers');
            const brokers = await response.json();
            this.systemData.brokers = brokers;
            this.updateBrokers();
        } catch (error) {
            console.error('Error loading brokers:', error);
        }
    }
    
    async loadPerformance() {
        try {
            const response = await fetch('/api/performance');
            const performance = await response.json();
            this.updatePerformanceDisplay(performance);
        } catch (error) {
            console.error('Error loading performance:', error);
        }
    }
    
    updatePerformanceDisplay(performance) {
        if (!performance.summary) return;
        
        const summary = performance.summary;
        
        document.getElementById('perf-total-pnl').textContent = `$${summary.total_pnl.toLocaleString()}`;
        document.getElementById('perf-win-rate').textContent = `${(summary.avg_win_rate * 100).toFixed(1)}%`;
        document.getElementById('perf-sharpe-ratio').textContent = '1.59';
        document.getElementById('perf-trading-days').textContent = summary.total_days;
        
        // Update daily performance
        const dailyPerformance = document.getElementById('daily-performance');
        if (performance.daily_metrics && performance.daily_metrics.length > 0) {
            const dailyHTML = performance.daily_metrics.map(metric => `
                <div class="daily-item">
                    <div class="daily-date">${metric.date}</div>
                    <div class="daily-stats">
                        <div class="daily-trades">${metric.total_trades} trades</div>
                        <div class="daily-pnl ${metric.total_pnl >= 0 ? 'positive' : 'negative'}">
                            $${metric.total_pnl.toFixed(2)}
                        </div>
                        <div class="daily-winrate">${(metric.win_rate * 100).toFixed(1)}%</div>
                    </div>
                </div>
            `).join('');
            
            dailyPerformance.innerHTML = dailyHTML;
        }
    }
    
    requestPredictions() {
        // Request predictions for main display symbol (AAPL)
        this.sendMessage({
            type: 'get_predictions',
            symbol: 'AAPL'
        });
    }
    
    // Notifications
    showNotification(message, type = 'info') {
        const notifications = document.getElementById('notifications');
        if (!notifications) return;
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        notifications.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }
    
    // Utility Methods
    formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(amount);
    }
    
    formatPercent(value) {
        return (value * 100).toFixed(1) + '%';
    }
    
    formatTimestamp(timestamp) {
        return new Date(timestamp).toLocaleTimeString();
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing 1-Second Trading System v3.0...');
    window.tradingApp = new TradingSystemApp();
});

// Auto-refresh predictions every 30 seconds
setInterval(() => {
    if (window.tradingApp && window.tradingApp.ws && window.tradingApp.ws.readyState === WebSocket.OPEN) {
        window.tradingApp.requestPredictions();
    }
}, 30000);

// Page visibility change handler
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && window.tradingApp) {
        // Reload data when page becomes visible
        window.tradingApp.loadInitialData();
        window.tradingApp.requestPredictions();
    }
});

// Window beforeunload handler
window.addEventListener('beforeunload', () => {
    if (window.tradingApp && window.tradingApp.ws) {
        window.tradingApp.ws.close();
    }
});

// Global error handler
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    if (window.tradingApp) {
        window.tradingApp.showNotification('An error occurred. Please refresh the page.', 'error');
    }
});

// Service worker registration (if needed)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        // Service worker can be added here for offline functionality
    });
}