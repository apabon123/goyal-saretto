from AlgorithmImports import *
import numpy as np
from datetime import timedelta
from collections import defaultdict, deque

class GoyalSarettoStrategy(QCAlgorithm):
    """
    Goyal-Saretto volatility strategy implementation
    Simplified for better performance and data handling
    """
    
    def Initialize(self):
        # Set backtest period
        self.SetStartDate(2021, 1, 1)
        self.SetEndDate(2022, 6, 30)
        self.SetCash(1000000)
        
        # Store starting portfolio value
        self.starting_portfolio_value = self.Portfolio.TotalPortfolioValue
        
        # Reduced warmup period for faster initialization
        self.SetWarmup(timedelta(days=280))  # Just enough for 252 trading days + buffer
        
        # Focus on highly liquid stocks with active options
        self.stock_symbols = [
            "AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ", "AMZN", "NVDA", 
            "META", "NFLX", "AMD", "JPM", "BAC", "WMT", "DIS"
        ]
        
        # Strategy parameters
        self.lookback_period = 252  # 252 trading days for HV
        self.min_trading_days = 200  # Reduced minimum for faster startup
        self.atm_moneyness_range = (0.95, 1.05)  # Wider range for better data availability
        self.expiration_range = (10, 50)  # Slightly wider range
        self.portfolio_allocation = 0.20  # 20% allocation
        self.max_position_size = 0.10   # Max 10% per position
        self.min_hv_iv_threshold = 0.02  # Higher threshold for better signals
        
        # Simplified data structures
        self.price_history = {}
        self.return_history = {}
        self.volatility_data = {}
        self.current_positions = {}
        
        # Track valid symbols that have sufficient data
        self.valid_symbols = []
        self.symbols_added = []
        
        # Add equities and options
        for symbol in self.stock_symbols:
            try:
                equity = self.AddEquity(symbol, Resolution.Daily)
                equity.SetDataNormalizationMode(DataNormalizationMode.Raw)
                
                # Add options
                option = self.AddOption(symbol)
                option.SetFilter(-20, 20, timedelta(5), timedelta(60))
                
                # Initialize data structures
                symbol_obj = equity.Symbol
                self.price_history[symbol_obj] = deque(maxlen=300)
                self.return_history[symbol_obj] = deque(maxlen=260)
                self.volatility_data[symbol_obj] = {
                    'hv': None,
                    'iv': None,
                    'hv_iv_diff': None,
                    'last_update': None
                }
                
                self.symbols_added.append(symbol)
                self.Log(f"Added {symbol}")
                
            except Exception as e:
                self.Log(f"Failed to add {symbol}: {e}")
        
        # Performance tracking
        self.trade_count = 0
        self.last_rebalance = None
        self.rebalance_count = 0
        
        # Schedule monthly rebalancing
        self.Schedule.On(
            self.DateRules.MonthEnd(),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self.Rebalance
        )
        
        # Daily data updates
        self.Schedule.On(
            self.DateRules.EveryDay("SPY"),
            self.TimeRules.AfterMarketOpen("SPY", 5),
            self.UpdateData
        )
        
        self.Log(f"Strategy initialized with {len(self.symbols_added)} symbols")
    
    def UpdateData(self):
        """Update price and return history"""
        if self.IsWarmingUp:
            # During warmup, just collect price data
            for symbol_str in self.symbols_added:
                symbol_obj = Symbol.Create(symbol_str, SecurityType.Equity, Market.USA)
                
                if symbol_obj in self.CurrentSlice.Bars:
                    price = self.CurrentSlice.Bars[symbol_obj].Close
                    
                    if price > 0:
                        prev_price = self.price_history[symbol_obj][-1] if len(self.price_history[symbol_obj]) > 0 else price
                        self.price_history[symbol_obj].append(price)
                        
                        # Calculate return
                        if prev_price > 0 and prev_price != price:
                            daily_return = np.log(price / prev_price)
                            if abs(daily_return) < 0.3:  # Filter extreme moves
                                self.return_history[symbol_obj].append(daily_return)
            return
        
        # After warmup, update data and check quality
        valid_count = 0
        for symbol_str in self.symbols_added:
            symbol_obj = Symbol.Create(symbol_str, SecurityType.Equity, Market.USA)
            
            if symbol_obj in self.CurrentSlice.Bars:
                price = self.CurrentSlice.Bars[symbol_obj].Close
                
                if price > 0:
                    prev_price = self.price_history[symbol_obj][-1] if len(self.price_history[symbol_obj]) > 0 else price
                    self.price_history[symbol_obj].append(price)
                    
                    # Calculate return
                    if prev_price > 0 and prev_price != price:
                        daily_return = np.log(price / prev_price)
                        if abs(daily_return) < 0.3:  # Filter extreme moves
                            self.return_history[symbol_obj].append(daily_return)
                    
                    # Check if we have enough data
                    if len(self.return_history[symbol_obj]) >= self.min_trading_days:
                        if symbol_str not in self.valid_symbols:
                            self.valid_symbols.append(symbol_str)
                            self.Log(f"Added {symbol_str} to valid symbols with {len(self.return_history[symbol_obj])} returns")
                        valid_count += 1
        
        # Log status periodically
        if self.Time.day == 1:  # First day of month
            self.Log(f"Valid symbols: {len(self.valid_symbols)}, Total data points collected")
    
    def Rebalance(self):
        """Main rebalancing logic"""
        if self.IsWarmingUp:
            return
        
        self.rebalance_count += 1
        self.Log(f"\n=== REBALANCING {self.Time} (#{self.rebalance_count}) ===")
        
        # Calculate volatility signals
        signals = self.CalculateVolatilitySignals()
        
        self.Log(f"Calculated {len(signals)} volatility signals from {len(self.valid_symbols)} valid symbols")
        
        if len(signals) < 2:
            self.Log(f"Insufficient signals: {len(signals)} (minimum 2 required)")
            return
        
        # Close existing positions
        self.CloseAllOptionPositions()
        
        # Sort by HV-IV difference
        signals.sort(key=lambda x: x['hv_iv_diff'], reverse=True)
        
        # Select top and bottom signals
        n_positions = min(4, len(signals))
        n_long = n_positions // 2
        n_short = n_positions // 2
        
        long_signals = signals[:n_long] if n_long > 0 else []
        short_signals = signals[-n_short:] if n_short > 0 else []
        
        # Filter by threshold
        long_signals = [s for s in long_signals if s['hv_iv_diff'] > self.min_hv_iv_threshold]
        short_signals = [s for s in short_signals if s['hv_iv_diff'] < -self.min_hv_iv_threshold]
        
        total_positions = len(long_signals) + len(short_signals)
        if total_positions == 0:
            self.Log("No signals meet threshold criteria")
            return
        
        # Position sizing
        position_size = min(self.portfolio_allocation / total_positions, self.max_position_size)
        
        # Execute trades
        long_opened = 0
        short_opened = 0
        
        for signal in long_signals:
            if self.OpenStraddlePosition(signal['symbol'], position_size, True):
                long_opened += 1
                self.Log(f"LONG: {signal['symbol_str']} HV={signal['hv']:.3f} IV={signal['iv']:.3f} Diff={signal['hv_iv_diff']:.3f}")
        
        for signal in short_signals:
            if self.OpenStraddlePosition(signal['symbol'], position_size, False):
                short_opened += 1
                self.Log(f"SHORT: {signal['symbol_str']} HV={signal['hv']:.3f} IV={signal['iv']:.3f} Diff={signal['hv_iv_diff']:.3f}")
        
        self.Log(f"Opened {long_opened} long and {short_opened} short positions")
        self.Log("=== END REBALANCING ===\n")
        
        self.last_rebalance = self.Time
    
    def CalculateVolatilitySignals(self):
        """Calculate HV and IV for all valid symbols"""
        signals = []
        
        for symbol_str in self.valid_symbols:
            symbol_obj = Symbol.Create(symbol_str, SecurityType.Equity, Market.USA)
            
            try:
                # Calculate HV
                hv = self.CalculateHistoricalVolatility(symbol_obj)
                if hv is None:
                    continue
                
                # Calculate IV
                iv = self.CalculateImpliedVolatility(symbol_obj)
                if iv is None:
                    continue
                
                # Store and create signal
                self.volatility_data[symbol_obj].update({
                    'hv': hv,
                    'iv': iv,
                    'hv_iv_diff': hv - iv,
                    'last_update': self.Time
                })
                
                signals.append({
                    'symbol': symbol_obj,
                    'symbol_str': symbol_str,
                    'hv': hv,
                    'iv': iv,
                    'hv_iv_diff': hv - iv
                })
                
            except Exception as e:
                self.Debug(f"Error calculating volatility for {symbol_str}: {e}")
        
        return signals
    
    def CalculateHistoricalVolatility(self, symbol):
        """Calculate annualized historical volatility"""
        if symbol not in self.return_history:
            return None
        
        returns = list(self.return_history[symbol])
        
        if len(returns) < self.min_trading_days:
            return None
        
        # Use last 252 returns or available returns
        recent_returns = returns[-min(self.lookback_period, len(returns)):]
        
        try:
            # Filter out invalid returns
            valid_returns = [r for r in recent_returns if not (np.isnan(r) or np.isinf(r))]
            
            if len(valid_returns) < self.min_trading_days * 0.8:  # Need at least 80% valid data
                return None
            
            # Calculate volatility
            daily_vol = np.std(valid_returns, ddof=1)
            annual_vol = daily_vol * np.sqrt(252)
            
            # Validate result
            if 0.05 < annual_vol < 3.0:  # Reasonable bounds
                return float(annual_vol)
            
        except Exception as e:
            self.Debug(f"HV calculation error for {symbol.Value}: {e}")
        
        return None
    
    def CalculateImpliedVolatility(self, symbol):
        """Calculate average implied volatility from ATM options"""
        option_symbol = Symbol.CreateCanonicalOption(symbol)
        option_chain = self.CurrentSlice.OptionChains.get(option_symbol)
        
        if not option_chain or symbol not in self.CurrentSlice.Bars:
            return None
        
        try:
            underlying_price = self.CurrentSlice.Bars[symbol].Close
            if underlying_price <= 0:
                return None
            
            valid_ivs = []
            
            for option in option_chain:
                # Check expiration range
                days_to_expiry = (option.Expiry.date() - self.Time.date()).days
                if not (self.expiration_range[0] <= days_to_expiry <= self.expiration_range[1]):
                    continue
                
                # Check moneyness
                moneyness = option.Strike / underlying_price
                if not (self.atm_moneyness_range[0] <= moneyness <= self.atm_moneyness_range[1]):
                    continue
                
                # Validate IV
                if (hasattr(option, 'ImpliedVolatility') and 
                    0.05 < option.ImpliedVolatility < 3.0 and
                    option.BidPrice > 0 and option.AskPrice > option.BidPrice):
                    valid_ivs.append(option.ImpliedVolatility)
            
            if len(valid_ivs) >= 2:  # Need at least 2 valid options
                return float(np.mean(valid_ivs))
            
        except Exception as e:
            self.Debug(f"IV calculation error for {symbol.Value}: {e}")
        
        return None
    
    def OpenStraddlePosition(self, symbol, allocation, is_long):
        """Open a straddle position"""
        call, put = self.GetATMOptions(symbol)
        
        if not call or not put:
            return False
        
        try:
            current_value = self.Portfolio.TotalPortfolioValue
            dollar_allocation = current_value * allocation
            
            # Get prices
            call_price = (call.BidPrice + call.AskPrice) / 2
            put_price = (put.BidPrice + put.AskPrice) / 2
            straddle_price = call_price + put_price
            
            if straddle_price <= 0:
                return False
            
            # Calculate contracts
            contracts = max(1, int(dollar_allocation / (straddle_price * 100)))
            contracts = min(contracts, 10)  # Limit size
            
            # Execute trades
            if is_long:
                self.MarketOrder(call.Symbol, contracts)
                self.MarketOrder(put.Symbol, contracts)
            else:
                self.MarketOrder(call.Symbol, -contracts)
                self.MarketOrder(put.Symbol, -contracts)
            
            self.current_positions[symbol] = {
                'type': 'long_straddle' if is_long else 'short_straddle',
                'size': contracts,
                'call': call.Symbol,
                'put': put.Symbol,
                'entry_time': self.Time
            }
            
            self.trade_count += 1
            return True
            
        except Exception as e:
            self.Log(f"Error opening straddle for {symbol.Value}: {e}")
            return False
    
    def GetATMOptions(self, symbol):
        """Get closest ATM call and put options"""
        option_symbol = Symbol.CreateCanonicalOption(symbol)
        option_chain = self.CurrentSlice.OptionChains.get(option_symbol)
        
        if not option_chain or symbol not in self.CurrentSlice.Bars:
            return None, None
        
        underlying_price = self.CurrentSlice.Bars[symbol].Close
        
        valid_calls = []
        valid_puts = []
        
        for option in option_chain:
            days_to_expiry = (option.Expiry.date() - self.Time.date()).days
            if not (self.expiration_range[0] <= days_to_expiry <= self.expiration_range[1]):
                continue
            
            moneyness = option.Strike / underlying_price
            if not (self.atm_moneyness_range[0] <= moneyness <= self.atm_moneyness_range[1]):
                continue
            
            if (option.BidPrice > 0 and option.AskPrice > option.BidPrice and
                hasattr(option, 'ImpliedVolatility') and option.ImpliedVolatility > 0):
                
                if option.Right == OptionRight.Call:
                    valid_calls.append(option)
                else:
                    valid_puts.append(option)
        
        best_call = min(valid_calls, key=lambda x: abs(x.Strike - underlying_price)) if valid_calls else None
        best_put = min(valid_puts, key=lambda x: abs(x.Strike - underlying_price)) if valid_puts else None
        
        return best_call, best_put
    
    def CloseAllOptionPositions(self):
        """Close all option positions"""
        closed_count = 0
        for kvp in self.Securities:
            symbol = kvp.Key
            if symbol.SecurityType == SecurityType.Option:
                quantity = self.Portfolio[symbol].Quantity
                if quantity != 0:
                    self.Liquidate(symbol)
                    closed_count += 1
        
        self.current_positions.clear()
        if closed_count > 0:
            self.Log(f"Closed {closed_count} option positions")
    
    def OnOrderEvent(self, orderEvent):
        """Handle order events"""
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug(f"Order filled: {orderEvent.Symbol} {orderEvent.Direction} {orderEvent.FillQuantity}")
        elif orderEvent.Status in [OrderStatus.Invalid, OrderStatus.Canceled]:
            self.Log(f"Order failed: {orderEvent.Symbol} - {orderEvent.Status}")
    
    def OnEndOfAlgorithm(self):
        """Performance summary"""
        final_value = self.Portfolio.TotalPortfolioValue
        total_return = (final_value - self.starting_portfolio_value) / self.starting_portfolio_value
        
        self.Log(f"\n=== FINAL PERFORMANCE ===")
        self.Log(f"Starting Value: ${self.starting_portfolio_value:,.2f}")
        self.Log(f"Final Value: ${final_value:,.2f}")
        self.Log(f"Total Return: {total_return:.2%}")
        self.Log(f"Total Trades: {self.trade_count}")
        self.Log(f"Rebalances: {self.rebalance_count}")
        self.Log(f"Valid Symbols: {len(self.valid_symbols)}")
        
        # Show final volatility data
        for symbol_str in self.valid_symbols[:5]:  # Show first 5
            symbol_obj = Symbol.Create(symbol_str, SecurityType.Equity, Market.USA)
            if symbol_obj in self.volatility_data and self.volatility_data[symbol_obj]['hv']:
                data = self.volatility_data[symbol_obj]
                self.Log(f"{symbol_str}: HV={data['hv']:.3f} IV={data['iv']:.3f} Diff={data['hv_iv_diff']:.3f}")
        
        self.Log("=== END ALGORITHM ===")
