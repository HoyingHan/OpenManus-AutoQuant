"""Stock analysis tool for technical analysis and market insights."""

import json
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import talib
from scipy import stats

from app.tool.base import BaseTool, ToolResult


class StockAnalysisTool(BaseTool):
    """A tool for analyzing stock data and generating insights."""

    name: str = "stock_analysis"
    description: str = (
        "Analyzes stock data to generate technical insights, market analysis and trading opportunities. "
        "Can calculate technical indicators, identify market patterns, and generate analytical reports."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Analysis command to execute",
                "enum": [
                    "analyze_technicals",
                    "analyze_market_condition",
                    "generate_stock_report",
                    "calculate_indicators",
                ],
            },
            "data": {
                "type": "object",
                "description": "Stock price data as a JSON object with OHLCV structure",
            },
            "stock_code": {
                "type": "string",
                "description": "Stock code for the analysis",
            },
            "indicators": {
                "type": "array",
                "description": "List of technical indicators to calculate",
                "items": {
                    "type": "string",
                },
            },
            "period": {
                "type": "integer",
                "description": "Period for calculating technical indicators",
            },
        },
        "required": ["command"],
    }

    async def execute(
        self,
        command: str,
        data: Optional[Dict] = None,
        stock_code: Optional[str] = None,
        indicators: Optional[List[str]] = None,
        period: int = 14,
        **kwargs,
    ) -> ToolResult:
        """Execute the stock analysis command."""

        try:
            if command == "analyze_technicals":
                if not data:
                    return ToolResult(error="Stock data is required for analyze_technicals command")

                result = self._analyze_technicals(data, period)
                return result

            elif command == "analyze_market_condition":
                if not data:
                    return ToolResult(error="Stock data is required for analyze_market_condition command")

                result = self._analyze_market_condition(data)
                return result

            elif command == "generate_stock_report":
                if not data or not stock_code:
                    return ToolResult(error="Stock data and stock code are required for generate_stock_report command")

                result = self._generate_stock_report(data, stock_code)
                return result

            elif command == "calculate_indicators":
                if not data or not indicators:
                    return ToolResult(error="Stock data and indicators are required for calculate_indicators command")

                result = self._calculate_indicators(data, indicators, period)
                return result

            else:
                return ToolResult(error=f"Unknown command: {command}")

        except Exception as e:
            return ToolResult(error=f"Error executing command {command}: {str(e)}")

    def _preprocess_data(self, data: Dict) -> pd.DataFrame:
        """Convert JSON data to pandas DataFrame with proper structure."""
        # Check if data is already a DataFrame
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            # Convert to DataFrame if it's a dict with sample data
            if "summary" in data and "data_sample" in data["summary"]:
                df = pd.DataFrame(data["summary"]["data_sample"])
            elif "data_sample" in data:
                df = pd.DataFrame(data["data_sample"])
            else:
                # Assume it's a directly usable data structure
                df = pd.DataFrame(data if isinstance(data, list) else [data])

        # Standardize column names
        column_mapping = {
            '日期': 'date', '开盘': 'open', '收盘': 'close',
            '最高': 'high', '最低': 'low', '成交量': 'volume',
            'date': 'date', 'open': 'open', 'close': 'close',
            'high': 'high', 'low': 'low', 'volume': 'volume'
        }

        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        # Ensure required columns exist
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Data is missing required columns: {missing_columns}")

        # Convert date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            try:
                # 检查是否有嵌套列表并尝试提取
                df['date'] = df['date'].apply(lambda x: x[0] if isinstance(x, list) else x)
                df['date'] = pd.to_datetime(df['date'])
            except Exception as e:
                # 如果转换失败，创建一个基本的日期索引
                 print(f"日期转换错误：{e}，使用默认索引代替")
                 df['date'] = pd.date_range(start='2025-01-01', periods=len(df), freq='D')


        # Sort by date
        df = df.sort_values('date')

        # Ensure numeric types for price and volume columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def _calculate_indicators(
        self,
        data: Dict,
        indicators: List[str],
        period: int = 14
    ) -> ToolResult:
        """Calculate technical indicators for the given stock data."""
        try:
            df = self._preprocess_data(data)

            # Dictionary to store calculated indicators
            indicator_results = {}

            # Calculate requested indicators
            for indicator in indicators:
                if indicator.upper() == 'SMA':
                    indicator_results['SMA'] = talib.SMA(df['close'].values, timeperiod=period).tolist()

                elif indicator.upper() == 'EMA':
                    indicator_results['EMA'] = talib.EMA(df['close'].values, timeperiod=period).tolist()

                elif indicator.upper() == 'RSI':
                    indicator_results['RSI'] = talib.RSI(df['close'].values, timeperiod=period).tolist()

                elif indicator.upper() == 'MACD':
                    macd, macd_signal, macd_hist = talib.MACD(
                        df['close'].values,
                        fastperiod=12,
                        slowperiod=26,
                        signalperiod=9
                    )
                    indicator_results['MACD'] = {
                        'macd': macd.tolist(),
                        'signal': macd_signal.tolist(),
                        'histogram': macd_hist.tolist()
                    }

                elif indicator.upper() == 'BBANDS':
                    upper, middle, lower = talib.BBANDS(
                        df['close'].values,
                        timeperiod=period,
                        nbdevup=2,
                        nbdevdn=2,
                        matype=0
                    )
                    indicator_results['BBANDS'] = {
                        'upper': upper.tolist(),
                        'middle': middle.tolist(),
                        'lower': lower.tolist()
                    }

                elif indicator.upper() == 'ATR':
                    indicator_results['ATR'] = talib.ATR(
                        df['high'].values,
                        df['low'].values,
                        df['close'].values,
                        timeperiod=period
                    ).tolist()

                elif indicator.upper() == 'ADX':
                    indicator_results['ADX'] = talib.ADX(
                        df['high'].values,
                        df['low'].values,
                        df['close'].values,
                        timeperiod=period
                    ).tolist()

                elif indicator.upper() == 'VOLUME_MA':
                    indicator_results['VOLUME_MA'] = talib.SMA(df['volume'].values, timeperiod=period).tolist()

                else:
                    # If indicator is not supported, add a message
                    indicator_results[indicator] = f"Indicator {indicator} not supported"

            # Create a combined dataframe with date and indicators for the result
            result_df = pd.DataFrame()
            result_df['date'] = df['date'].astype(str)  # Convert dates to strings for JSON serialization

            # Add each indicator to the result dataframe
            for indicator, values in indicator_results.items():
                if isinstance(values, list):
                    # Handle simple indicators
                    result_df[indicator] = values
                elif isinstance(values, dict):
                    # Handle compound indicators like MACD and BBANDS
                    for key, val in values.items():
                        result_df[f"{indicator}_{key}"] = val

            # Convert to records for JSON serialization
            result_data = result_df.tail(30).to_dict(orient='records')  # Show just last 30 records for readability

            return ToolResult(output=json.dumps({
                'indicators_calculated': list(indicator_results.keys()),
                'period': period,
                'data_points': len(df),
                'recent_data': result_data
            }, ensure_ascii=False, indent=2))

        except Exception as e:
            return ToolResult(error=f"Error calculating indicators: {str(e)}")

    def _analyze_technicals(self, data: Dict, period: int = 14) -> ToolResult:
        """Analyze technical indicators and patterns for the given stock data."""
        try:
            df = self._preprocess_data(data)

            # Calculate key technical indicators
            # RSI
            df['rsi'] = talib.RSI(df['close'].values, timeperiod=period)

            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                df['close'].values,
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )

            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                df['close'].values,
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2,
                matype=0
            )

            # Moving Averages
            df['sma_20'] = talib.SMA(df['close'].values, timeperiod=20)
            df['sma_50'] = talib.SMA(df['close'].values, timeperiod=50)
            df['sma_200'] = talib.SMA(df['close'].values, timeperiod=200)

            # ATR for volatility
            df['atr'] = talib.ATR(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                timeperiod=14
            )

            # Trend analysis
            is_uptrend = df['close'].iloc[-1] > df['sma_50'].iloc[-1] > df['sma_200'].iloc[-1]
            is_downtrend = df['close'].iloc[-1] < df['sma_50'].iloc[-1] < df['sma_200'].iloc[-1]

            # Momentum analysis
            rsi_current = df['rsi'].iloc[-1] if not np.isnan(df['rsi'].iloc[-1]) else 50
            is_overbought = rsi_current > 70
            is_oversold = rsi_current < 30

            # MACD analysis
            macd_current = df['macd'].iloc[-1] if not np.isnan(df['macd'].iloc[-1]) else 0
            macd_signal_current = df['macd_signal'].iloc[-1] if not np.isnan(df['macd_signal'].iloc[-1]) else 0
            macd_hist_current = df['macd_hist'].iloc[-1] if not np.isnan(df['macd_hist'].iloc[-1]) else 0
            is_macd_bullish = macd_current > macd_signal_current and macd_hist_current > 0
            is_macd_bearish = macd_current < macd_signal_current and macd_hist_current < 0

            # Volatility analysis
            avg_atr = df['atr'].iloc[-5:].mean() if not df['atr'].iloc[-5:].isna().all() else 0
            price_latest = df['close'].iloc[-1]
            volatility_pct = (avg_atr / price_latest) * 100 if price_latest > 0 else 0

            # Support and resistance levels
            # Simple implementation using recent highs and lows
            recent_df = df.tail(30)
            resistance_level = recent_df['high'].max()
            support_level = recent_df['low'].min()

            # Price relative to Bollinger Bands
            price_vs_bb = "MIDDLE"
            if not np.isnan(df['bb_upper'].iloc[-1]) and df['close'].iloc[-1] >= df['bb_upper'].iloc[-1]:
                price_vs_bb = "UPPER"
            elif not np.isnan(df['bb_lower'].iloc[-1]) and df['close'].iloc[-1] <= df['bb_lower'].iloc[-1]:
                price_vs_bb = "LOWER"

            # Compile analysis results
            analysis_results = {
                "trend": {
                    "is_uptrend": bool(is_uptrend),
                    "is_downtrend": bool(is_downtrend),
                    "sma_20": round(float(df['sma_20'].iloc[-1]), 2) if not np.isnan(df['sma_20'].iloc[-1]) else None,
                    "sma_50": round(float(df['sma_50'].iloc[-1]), 2) if not np.isnan(df['sma_50'].iloc[-1]) else None,
                    "sma_200": round(float(df['sma_200'].iloc[-1]), 2) if not np.isnan(df['sma_200'].iloc[-1]) else None,
                    "current_price": round(float(df['close'].iloc[-1]), 2),
                },
                "momentum": {
                    "rsi": round(float(rsi_current), 2),
                    "is_overbought": bool(is_overbought),
                    "is_oversold": bool(is_oversold),
                    "macd": round(float(macd_current), 2) if not np.isnan(macd_current) else None,
                    "macd_signal": round(float(macd_signal_current), 2) if not np.isnan(macd_signal_current) else None,
                    "macd_histogram": round(float(macd_hist_current), 2) if not np.isnan(macd_hist_current) else None,
                    "is_macd_bullish": bool(is_macd_bullish),
                    "is_macd_bearish": bool(is_macd_bearish),
                },
                "volatility": {
                    "atr": round(float(avg_atr), 2),
                    "volatility_percentage": round(volatility_pct, 2),
                },
                "support_resistance": {
                    "support_level": round(float(support_level), 2),
                    "resistance_level": round(float(resistance_level), 2),
                    "price_to_support_pct": round(((df['close'].iloc[-1] / support_level) - 1) * 100, 2),
                    "price_to_resistance_pct": round(((resistance_level / df['close'].iloc[-1]) - 1) * 100, 2),
                },
                "bollinger_bands": {
                    "upper_band": round(float(df['bb_upper'].iloc[-1]), 2) if not np.isnan(df['bb_upper'].iloc[-1]) else None,
                    "middle_band": round(float(df['bb_middle'].iloc[-1]), 2) if not np.isnan(df['bb_middle'].iloc[-1]) else None,
                    "lower_band": round(float(df['bb_lower'].iloc[-1]), 2) if not np.isnan(df['bb_lower'].iloc[-1]) else None,
                    "price_relative_to_band": price_vs_bb,
                },
            }

            # Generate trading signals based on technical analysis
            signals = []

            if is_uptrend and is_macd_bullish and not is_overbought:
                signals.append({
                    "signal": "BULLISH",
                    "strength": "STRONG",
                    "reason": "Uptrend with bullish MACD and RSI not overbought"
                })
            elif is_uptrend and not is_macd_bearish:
                signals.append({
                    "signal": "BULLISH",
                    "strength": "MODERATE",
                    "reason": "Uptrend but MACD not showing strong bullish momentum"
                })
            elif is_oversold and df['close'].iloc[-1] >= df['close'].iloc[-2]:
                signals.append({
                    "signal": "BULLISH",
                    "strength": "WEAK",
                    "reason": "Oversold conditions with potential price reversal"
                })

            if is_downtrend and is_macd_bearish and not is_oversold:
                signals.append({
                    "signal": "BEARISH",
                    "strength": "STRONG",
                    "reason": "Downtrend with bearish MACD and RSI not oversold"
                })
            elif is_downtrend and not is_macd_bullish:
                signals.append({
                    "signal": "BEARISH",
                    "strength": "MODERATE",
                    "reason": "Downtrend but MACD not showing strong bearish momentum"
                })
            elif is_overbought and df['close'].iloc[-1] <= df['close'].iloc[-2]:
                signals.append({
                    "signal": "BEARISH",
                    "strength": "WEAK",
                    "reason": "Overbought conditions with potential price reversal"
                })

            if price_vs_bb == "UPPER" and is_overbought:
                signals.append({
                    "signal": "BEARISH",
                    "strength": "MODERATE",
                    "reason": "Price at upper Bollinger Band with overbought RSI"
                })
            elif price_vs_bb == "LOWER" and is_oversold:
                signals.append({
                    "signal": "BULLISH",
                    "strength": "MODERATE",
                    "reason": "Price at lower Bollinger Band with oversold RSI"
                })

            # If no clear signals, suggest neutral
            if not signals:
                signals.append({
                    "signal": "NEUTRAL",
                    "strength": "MODERATE",
                    "reason": "No clear technical signals at current price levels"
                })

            analysis_results["signals"] = signals

            return ToolResult(output=json.dumps(analysis_results, ensure_ascii=False, indent=2))

        except Exception as e:
            return ToolResult(error=f"Error analyzing technicals: {str(e)}")

    def _analyze_market_condition(self, data: Dict) -> ToolResult:
        """Analyze the overall market condition based on the stock data."""
        try:
            df = self._preprocess_data(data)

            # Calculate market metrics
            # 1. Trend strength
            df['returns'] = df['close'].pct_change()
            df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1

            # 2. Volatility
            volatility_20d = df['returns'].rolling(window=20).std() * np.sqrt(252)  # Annualized
            current_volatility = volatility_20d.iloc[-1] if not volatility_20d.iloc[-1:].isna().all() else 0

            # 3. Momentum
            momentum_20d = df['returns'].rolling(window=20).mean() * 252  # Annualized
            current_momentum = momentum_20d.iloc[-1] if not momentum_20d.iloc[-1:].isna().all() else 0

            # 4. Market efficiency ratio (MER)
            df['direction'] = np.where(df['close'].diff() > 0, 1, -1)
            df['direction'] = np.where(df['close'].diff() == 0, 0, df['direction'])
            mer_period = min(20, len(df) - 1)
            direction_changes = sum(abs(df['direction'].diff().tail(mer_period).fillna(0)))
            mer = 1 - (direction_changes / mer_period) if mer_period > 0 else 0

            # 5. Recent performance
            returns_1m = df['cumulative_returns'].iloc[-1] - df['cumulative_returns'].iloc[-21] if len(df) > 21 else df['cumulative_returns'].iloc[-1]
            returns_3m = df['cumulative_returns'].iloc[-1] - df['cumulative_returns'].iloc[-63] if len(df) > 63 else returns_1m

            # 6. Trading volume analysis
            recent_volume = df['volume'].tail(5).mean()
            volume_20d_avg = df['volume'].rolling(window=20).mean().iloc[-1] if len(df) > 20 else df['volume'].mean()
            volume_trend = "INCREASING" if recent_volume > volume_20d_avg * 1.1 else "DECREASING" if recent_volume < volume_20d_avg * 0.9 else "STABLE"

            # 7. Price trend
            df['sma_20'] = talib.SMA(df['close'].values, timeperiod=20)
            df['sma_50'] = talib.SMA(df['close'].values, timeperiod=50)
            trend = "BULLISH" if df['close'].iloc[-1] > df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1] else \
                   "BEARISH" if df['close'].iloc[-1] < df['sma_20'].iloc[-1] < df['sma_50'].iloc[-1] else \
                   "NEUTRAL"

            # Compile market analysis
            market_analysis = {
                "trend": {
                    "direction": trend,
                    "strength": "STRONG" if abs(current_momentum) > 0.2 else "MODERATE" if abs(current_momentum) > 0.1 else "WEAK",
                    "efficiency": round(mer, 2),
                },
                "performance": {
                    "1_month_return": round(returns_1m * 100, 2) if not np.isnan(returns_1m) else None,
                    "3_month_return": round(returns_3m * 100, 2) if not np.isnan(returns_3m) else None,
                    "annualized_momentum": round(current_momentum * 100, 2) if not np.isnan(current_momentum) else None,
                },
                "volatility": {
                    "current": round(current_volatility * 100, 2) if not np.isnan(current_volatility) else None,
                    "level": "HIGH" if current_volatility > 0.3 else "MEDIUM" if current_volatility > 0.15 else "LOW",
                },
                "volume": {
                    "trend": volume_trend,
                    "recent_avg": int(recent_volume),
                    "20d_avg": int(volume_20d_avg),
                }
            }

            # Market condition assessment
            if trend == "BULLISH" and market_analysis["volatility"]["level"] != "HIGH" and volume_trend != "DECREASING":
                market_condition = "FAVORABLE"
                market_description = "The market shows a bullish trend with manageable volatility and healthy trading volumes. Conditions appear favorable for trend-following strategies."
            elif trend == "BEARISH" and market_analysis["volatility"]["level"] == "HIGH":
                market_condition = "CHALLENGING"
                market_description = "The market is in a bearish trend with high volatility. Trading conditions are challenging, suggesting a more cautious approach."
            elif trend == "NEUTRAL" and market_analysis["volatility"]["level"] != "LOW":
                market_condition = "WAIT_AND_SEE"
                market_description = "The market lacks a clear direction with moderate volatility. A wait-and-see approach or range-bound trading strategies may be appropriate."
            else:
                market_condition = "MIXED"
                market_description = "Market conditions show mixed signals. Consider a balanced approach with risk management as a priority."

            market_analysis["overall_assessment"] = {
                "condition": market_condition,
                "description": market_description
            }

            return ToolResult(output=json.dumps(market_analysis, ensure_ascii=False, indent=2))

        except Exception as e:
            return ToolResult(error=f"Error analyzing market condition: {str(e)}")

    def _generate_stock_report(self, data: Dict, stock_code: str) -> ToolResult:
        """Generate a comprehensive stock analysis report."""
        try:
            # Get technical analysis
            tech_analysis = self._analyze_technicals(data)
            if tech_analysis.error:
                return tech_analysis

            tech_results = json.loads(tech_analysis.output)

            # Get market condition analysis
            market_analysis = self._analyze_market_condition(data)
            if market_analysis.error:
                return market_analysis

            market_results = json.loads(market_analysis.output)

            # Additional statistical analysis
            df = self._preprocess_data(data)

            # Return distribution analysis
            returns = df['close'].pct_change().dropna()

            # Basic stats
            mean_return = returns.mean()
            std_return = returns.std()
            skewness = stats.skew(returns.dropna())
            kurtosis = stats.kurtosis(returns.dropna())

            # Risk metrics
            sharpe_ratio = (mean_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0
            max_drawdown = self._calculate_max_drawdown(df['close'])

            # Trend analysis over different timeframes
            price_current = df['close'].iloc[-1]
            price_5d_ago = df['close'].iloc[-6] if len(df) > 5 else df['close'].iloc[0]
            price_20d_ago = df['close'].iloc[-21] if len(df) > 20 else df['close'].iloc[0]
            price_60d_ago = df['close'].iloc[-61] if len(df) > 60 else df['close'].iloc[0]

            change_5d = ((price_current / price_5d_ago) - 1) * 100
            change_20d = ((price_current / price_20d_ago) - 1) * 100
            change_60d = ((price_current / price_60d_ago) - 1) * 100

            # Correlation with market (assuming the provided data is market data)
            # This would be more meaningful if we had a market index to compare with

            # Compile report
            report = {
                "stock_code": stock_code,
                "data_period": f"{df['date'].min()} to {df['date'].max()}",
                "current_price": round(price_current, 2),
                "price_changes": {
                    "5_day": round(change_5d, 2),
                    "20_day": round(change_20d, 2),
                    "60_day": round(change_60d, 2)
                },
                "technical_analysis": tech_results,
                "market_condition": market_results,
                "risk_metrics": {
                    "daily_volatility": round(std_return * 100, 2),
                    "annualized_volatility": round(std_return * np.sqrt(252) * 100, 2),
                    "sharpe_ratio": round(sharpe_ratio, 2),
                    "max_drawdown_pct": round(max_drawdown * 100, 2),
                },
                "return_distribution": {
                    "mean_daily_return": round(mean_return * 100, 4),
                    "annualized_return": round(mean_return * 252 * 100, 2),
                    "skewness": round(skewness, 2),
                    "kurtosis": round(kurtosis, 2),
                    "normality": "The returns show " + (
                        "significant deviation from normal distribution" if abs(skewness) > 1 or abs(kurtosis) > 3
                        else "moderate deviation from normal distribution" if abs(skewness) > 0.5 or abs(kurtosis) > 1
                        else "approximate normal distribution"
                    )
                }
            }

            # Generate trading recommendation
            tech_signals = tech_results.get("signals", [])

            # Extract the primary signal
            primary_signal = None
            for signal in tech_signals:
                if signal.get("strength") in ["STRONG", "MODERATE"]:
                    primary_signal = signal
                    break

            if not primary_signal:
                primary_signal = tech_signals[0] if tech_signals else {"signal": "NEUTRAL", "strength": "WEAK", "reason": "Insufficient signals"}

            # Generate recommendation based on signal and market condition
            market_condition = market_results.get("overall_assessment", {}).get("condition", "MIXED")

            recommendation = {
                "position": "HOLD",  # Default
                "confidence": "LOW",
                "reasoning": "",
                "risk_level": "MEDIUM",
                "suggested_stop_loss_pct": 0,
                "suggested_take_profit_pct": 0,
            }

            if primary_signal["signal"] == "BULLISH":
                if market_condition in ["FAVORABLE", "MIXED"]:
                    recommendation["position"] = "BUY"
                    recommendation["confidence"] = "HIGH" if primary_signal["strength"] == "STRONG" and market_condition == "FAVORABLE" else "MEDIUM"
                    recommendation["reasoning"] = f"Bullish technical signals in a {market_condition.lower()} market environment"
                    recommendation["risk_level"] = "MEDIUM"
                    recommendation["suggested_stop_loss_pct"] = round(tech_results["volatility"]["atr"] / price_current * 200, 1)
                    recommendation["suggested_take_profit_pct"] = round(tech_results["volatility"]["atr"] / price_current * 300, 1)
                else:
                    recommendation["position"] = "HOLD"
                    recommendation["confidence"] = "MEDIUM"
                    recommendation["reasoning"] = "Bullish signals present but overall market conditions are not favorable"
                    recommendation["risk_level"] = "HIGH"

            elif primary_signal["signal"] == "BEARISH":
                if report["risk_metrics"]["max_drawdown_pct"] > 15:
                    recommendation["position"] = "SELL"
                    recommendation["confidence"] = "HIGH" if primary_signal["strength"] == "STRONG" else "MEDIUM"
                    recommendation["reasoning"] = "Bearish technical signals with significant historical drawdown risk"
                    recommendation["risk_level"] = "HIGH"
                else:
                    recommendation["position"] = "HOLD" if market_condition != "CHALLENGING" else "SELL"
                    recommendation["confidence"] = "MEDIUM"
                    recommendation["reasoning"] = f"Bearish signals in a {market_condition.lower()} market, but moderate historical drawdown"
                    recommendation["risk_level"] = "MEDIUM"

            else:  # NEUTRAL
                recommendation["position"] = "HOLD"
                recommendation["confidence"] = "MEDIUM"
                recommendation["reasoning"] = "Neutral market signals suggest holding current positions and awaiting clearer direction"
                recommendation["risk_level"] = "LOW"

            report["trading_recommendation"] = recommendation

            return ToolResult(output=json.dumps(report, ensure_ascii=False, indent=2))

        except Exception as e:
            return ToolResult(error=f"Error generating stock report: {str(e)}")

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate the maximum drawdown from peak to trough."""
        # Calculate the cumulative maximum of the price series
        rolling_max = prices.cummax()

        # Calculate the drawdown in percentage terms
        drawdown = (prices / rolling_max) - 1

        # Get the maximum drawdown
        max_drawdown = drawdown.min()

        return abs(max_drawdown) if not np.isnan(max_drawdown) else 0

# For testing
if __name__ == "__main__":
    import asyncio
    import json

    async def test():
        try:
            tool = StockAnalysisTool()

            # Mock data
            with open("test_stock_data.json", "r") as f:
                data = json.load(f)

            # Test technical analysis
            result = await tool.execute(command="analyze_technicals", data=data)
            print("Technical Analysis:")
            print(result.output)

            # Test market condition analysis
            result = await tool.execute(command="analyze_market_condition", data=data)
            print("\nMarket Condition Analysis:")
            print(result.output)

            # Test stock report generation
            result = await tool.execute(command="generate_stock_report", data=data, stock_code="600519")
            print("\nStock Report:")
            print(result.output)

        except Exception as e:
            print(f"Error in test: {str(e)}")

    asyncio.run(test())
