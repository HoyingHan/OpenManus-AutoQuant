"""Strategy generator tool for creating trading strategies based on stock analysis."""

import os
from typing import Dict, List, Optional, Tuple, Any
import datetime
import json
import re

from app.tool.base import BaseTool, ToolResult, save_to_workspace, generate_timestamp_id, ensure_workspace_dir
from app.logger import logger


class StrategyGeneratorTool(BaseTool):
    """A tool for generating backtrader trading strategies based on stock analysis."""

    name: str = "strategy_generator"
    description: str = (
        "Generates backtrader trading strategy code based on stock analysis results. "
        "Creates Python code for technical indicator strategies, pattern recognition, and more."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Strategy generation command to execute",
                "enum": [
                    "generate_strategy",
                    "get_strategy_template",
                    "list_available_strategies",
                ],
            },
            "strategy_type": {
                "type": "string",
                "description": "Type of strategy to generate",
                "enum": [
                    "dynamic",        # 根据分析结果动态生成策略（推荐）
                    "momentum",
                    "trend_following",
                    "mean_reversion",
                    "dual_ma_crossover",
                    "macd",
                    "rsi",
                    "bollinger_bands",
                    "custom",
                ],
            },
            "analysis_results": {
                "type": "object",
                "description": "Results from stock analysis to base the strategy on",
            },
            "risk_params": {
                "type": "object",
                "description": "Risk parameters for the strategy",
                "properties": {
                    "target_annual_return": {"type": "number"},
                    "max_drawdown": {"type": "number"},
                    "sharpe_ratio": {"type": "number"},
                    "profit_loss_ratio": {"type": "number"},
                },
            },
            "custom_params": {
                "type": "object",
                "description": "Custom parameters for strategy customization",
                "properties": {
                    "use_dynamic_generation": {"type": "boolean"},
                    "use_sma": {"type": "boolean"},
                    "use_rsi": {"type": "boolean"},
                    "use_macd": {"type": "boolean"},
                    "use_bbands": {"type": "boolean"},
                    "use_atr": {"type": "boolean"},
                    "fast_period": {"type": "integer"},
                    "slow_period": {"type": "integer"},
                    "rsi_period": {"type": "integer"},
                    "rsi_overbought": {"type": "integer"},
                    "rsi_oversold": {"type": "integer"},
                    "macd_fast": {"type": "integer"},
                    "macd_slow": {"type": "integer"},
                    "macd_signal": {"type": "integer"},
                    "bb_period": {"type": "integer"},
                    "bb_dev": {"type": "number"},
                    "atr_period": {"type": "integer"},
                    "atr_multiplier": {"type": "number"},
                },
            },
        },
        "required": ["command"],
    }

    async def execute(
        self,
        command: str,
        strategy_type: Optional[str] = None,
        analysis_results: Optional[Dict] = None,
        risk_params: Optional[Dict] = None,
        custom_params: Optional[Dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Execute the strategy generation command."""

        try:
            if command == "generate_strategy":
                if not strategy_type or not analysis_results:
                    return ToolResult(error="Strategy type and analysis results are required for generate_strategy command")

                result = self._generate_strategy(strategy_type, analysis_results, risk_params, custom_params)
                return result

            elif command == "get_strategy_template":
                if not strategy_type:
                    return ToolResult(error="Strategy type is required for get_strategy_template command")

                result = self._get_strategy_template(strategy_type)
                return result

            elif command == "list_available_strategies":
                result = self._list_available_strategies()
                return result

            else:
                return ToolResult(error=f"Unknown command: {command}")

        except Exception as e:
            return ToolResult(error=f"Error executing command {command}: {str(e)}")

    def _list_available_strategies(self) -> ToolResult:
        """List all available strategy types with descriptions."""
        strategies = {
            "dynamic": "智能动态策略，根据市场分析结果自动构建最适合的策略逻辑和参数组合",
            "momentum": "Strategy based on price momentum indicators like RSI and MACD, entering when momentum is strong",
            "trend_following": "Strategy that identifies and follows market trends using moving averages and trend indicators",
            "mean_reversion": "Strategy that capitalizes on price movements returning to the mean, using indicators like Bollinger Bands",
            "dual_ma_crossover": "Classic strategy using two moving averages of different periods to generate entry/exit signals",
            "macd": "Strategy based solely on MACD (Moving Average Convergence Divergence) signals",
            "rsi": "Strategy trading overbought and oversold conditions using the Relative Strength Index",
            "bollinger_bands": "Strategy trading price movements relative to Bollinger Bands",
            "custom": "Custom strategy combining multiple indicators based on market analysis"
        }

        return ToolResult(output=json.dumps({
            "available_strategies": strategies,
            "recommendation": "推荐使用'dynamic'策略类型，它能根据具体分析结果动态生成最优策略组合。"
        }, ensure_ascii=False, indent=2))

    def _get_strategy_template(self, strategy_type: str) -> ToolResult:
        """Get a template for the specified strategy type."""
        templates = {
            "momentum": self._momentum_strategy_template(),
            "trend_following": self._trend_following_strategy_template(),
            "mean_reversion": self._mean_reversion_strategy_template(),
            "dual_ma_crossover": self._dual_ma_crossover_strategy_template(),
            "macd": self._macd_strategy_template(),
            "rsi": self._rsi_strategy_template(),
            "bollinger_bands": self._bollinger_bands_strategy_template(),
            "custom": self._custom_strategy_template(),
        }

        if strategy_type not in templates:
            return ToolResult(error=f"Unknown strategy type: {strategy_type}")

        return ToolResult(output=json.dumps({
            "strategy_type": strategy_type,
            "template": templates[strategy_type],
            "usage_instructions": "This template can be customized with specific parameters based on stock analysis."
        }, ensure_ascii=False, indent=2))

    def _generate_strategy(
        self,
        strategy_type: str,
        analysis_results: Dict,
        risk_params: Optional[Dict] = None,
        custom_params: Optional[Dict] = None
    ) -> ToolResult:
        """
        根据分析结果和参数生成交易策略。
        生成的策略代码将保存到工作空间。

        参数:
            strategy_type: 策略类型
            analysis_results: 分析结果
            risk_params: 风险参数
            custom_params: 自定义参数

        返回:
            包含策略信息的ToolResult对象
        """
        from app.tool.base import save_to_workspace, generate_timestamp_id
        from app.logger import logger

        # 设置默认风险参数（如果未提供）
        if not risk_params:
            risk_params = {
                "target_annual_return": 0.15,  # 15%
                "max_drawdown": 0.20,  # 20%
                "sharpe_ratio": 1.0,
                "profit_loss_ratio": 1.5
            }

        # 处理分析结果并提取关键见解
        strategy_insights = self._extract_strategy_insights(analysis_results)

        # 新的动态策略生成方法
        if strategy_type == "dynamic" or custom_params and custom_params.get("use_dynamic_generation", False):
            logger.info("使用动态策略生成逻辑创建策略")
            strategy_code = self._generate_dynamic_strategy(strategy_insights, risk_params, custom_params, analysis_results)
        # 保留原有生成方法作为备选
        elif strategy_type == "momentum":
            strategy_code = self._generate_momentum_strategy(strategy_insights, risk_params, custom_params)
        elif strategy_type == "trend_following":
            strategy_code = self._generate_trend_following_strategy(strategy_insights, risk_params, custom_params)
        elif strategy_type == "mean_reversion":
            strategy_code = self._generate_mean_reversion_strategy(strategy_insights, risk_params, custom_params)
        elif strategy_type == "dual_ma_crossover":
            strategy_code = self._generate_dual_ma_crossover_strategy(strategy_insights, risk_params, custom_params)
        elif strategy_type == "macd":
            strategy_code = self._generate_macd_strategy(strategy_insights, risk_params, custom_params)
        elif strategy_type == "rsi":
            strategy_code = self._generate_rsi_strategy(strategy_insights, risk_params, custom_params)
        elif strategy_type == "bollinger_bands":
            strategy_code = self._generate_bollinger_bands_strategy(strategy_insights, risk_params, custom_params)
        elif strategy_type == "custom":
            strategy_code = self._generate_custom_strategy(strategy_insights, risk_params, custom_params)
        else:
            # 默认使用动态策略生成
            logger.info(f"未知策略类型 '{strategy_type}'，使用动态策略生成")
            strategy_code = self._generate_dynamic_strategy(strategy_insights, risk_params, custom_params, analysis_results)

        # 生成唯一ID用于策略文件
        strategy_id = generate_timestamp_id(f"{strategy_type}_strategy")
        strategy_filename = f"{strategy_id}.py"

        # 保存策略到工作空间
        try:
            strategy_file_path = save_to_workspace(strategy_code, strategy_filename, "strategies")
            logger.info(f"交易策略已保存到: {strategy_file_path}")

            # 保存策略元数据
            strategy_metadata = {
                "id": strategy_id,
                "strategy_type": strategy_type,
                "strategy_file": strategy_file_path,
                "risk_parameters": risk_params,
                "strategy_insights": strategy_insights,
                "custom_parameters": custom_params,
                "timestamp": datetime.datetime.now().isoformat(),
                "analysis_results_id": analysis_results.get("id", "unknown")
            }

            metadata_file = f"{strategy_id}_metadata.json"
            metadata_path = save_to_workspace(strategy_metadata, metadata_file, "strategies", is_json=True)
            logger.info(f"交易策略元数据已保存到: {metadata_path}")

        except Exception as e:
            logger.exception(f"保存策略文件时出错: {str(e)}")
            return ToolResult(error=f"保存策略文件时出错: {str(e)}")

        return ToolResult(output=json.dumps({
            "strategy_id": strategy_id,
            "strategy_type": strategy_type,
            "strategy_file": strategy_file_path,
            "metadata_file": metadata_path,
            "risk_parameters": risk_params,
            "strategy_insights": strategy_insights
        }, ensure_ascii=False, indent=2))

    def _extract_strategy_insights(self, analysis_results: Dict) -> Dict:
        """Extract key insights from analysis results for strategy generation."""
        insights = {
            "trend": {
                "direction": "neutral",
                "strength": "weak"
            },
            "momentum": {
                "state": "neutral",
                "strength": "weak"
            },
            "volatility": {
                "level": "medium",
                "atr_percentage": 2.0
            },
            "indicators": {
                "rsi_level": 50,
                "macd_signal": "neutral",
                "bollinger_position": "middle"
            },
            "support_resistance": {
                "support_level": None,
                "resistance_level": None,
                "support_distance_pct": 0,
                "resistance_distance_pct": 0
            }
        }

        # Process technical analysis results if available
        if "technical_analysis" in analysis_results:
            tech = analysis_results["technical_analysis"]

            # Extract trend information
            if "trend" in tech:
                if tech["trend"].get("is_uptrend", False):
                    insights["trend"]["direction"] = "bullish"
                    insights["trend"]["strength"] = "strong"
                elif tech["trend"].get("is_downtrend", False):
                    insights["trend"]["direction"] = "bearish"
                    insights["trend"]["strength"] = "strong"

            # Extract momentum information
            if "momentum" in tech:
                rsi = tech["momentum"].get("rsi", 50)
                insights["indicators"]["rsi_level"] = rsi

                if rsi > 70:
                    insights["momentum"]["state"] = "overbought"
                    insights["momentum"]["strength"] = "strong"
                elif rsi < 30:
                    insights["momentum"]["state"] = "oversold"
                    insights["momentum"]["strength"] = "strong"

                if tech["momentum"].get("is_macd_bullish", False):
                    insights["indicators"]["macd_signal"] = "bullish"
                elif tech["momentum"].get("is_macd_bearish", False):
                    insights["indicators"]["macd_signal"] = "bearish"

            # Extract volatility information
            if "volatility" in tech:
                atr_pct = tech["volatility"].get("volatility_percentage", 2.0)
                insights["volatility"]["atr_percentage"] = atr_pct

                if atr_pct > 3.0:
                    insights["volatility"]["level"] = "high"
                elif atr_pct < 1.0:
                    insights["volatility"]["level"] = "low"

            # Extract bollinger band information
            if "bollinger_bands" in tech:
                insights["indicators"]["bollinger_position"] = tech["bollinger_bands"].get("price_relative_to_band", "middle")

            # Extract support/resistance information
            if "support_resistance" in tech:
                insights["support_resistance"]["support_level"] = tech["support_resistance"].get("support_level")
                insights["support_resistance"]["resistance_level"] = tech["support_resistance"].get("resistance_level")
                insights["support_resistance"]["support_distance_pct"] = tech["support_resistance"].get("price_to_support_pct", 0)
                insights["support_resistance"]["resistance_distance_pct"] = tech["support_resistance"].get("price_to_resistance_pct", 0)

        # Process market condition if available
        if "market_condition" in analysis_results:
            market = analysis_results["market_condition"]

            # Adjust insights based on overall market condition
            if "overall_assessment" in market and "condition" in market["overall_assessment"]:
                condition = market["overall_assessment"]["condition"]

                if condition == "FAVORABLE":
                    if insights["trend"]["direction"] == "bullish":
                        insights["trend"]["strength"] = "strong"
                elif condition == "CHALLENGING":
                    if insights["trend"]["direction"] == "bearish":
                        insights["trend"]["strength"] = "strong"

        # Process trading recommendation if available
        if "trading_recommendation" in analysis_results:
            rec = analysis_results["trading_recommendation"]

            if rec.get("position") == "BUY":
                insights["recommended_action"] = "buy"
                insights["stop_loss_pct"] = rec.get("suggested_stop_loss_pct", 5)
                insights["take_profit_pct"] = rec.get("suggested_take_profit_pct", 10)
            elif rec.get("position") == "SELL":
                insights["recommended_action"] = "sell"
            else:
                insights["recommended_action"] = "hold"

        return insights

    def _generate_momentum_strategy(
        self,
        insights: Dict,
        risk_params: Dict,
        custom_params: Optional[Dict] = None
    ) -> str:
        """Generate a momentum-based trading strategy."""
        # Use custom params if provided, otherwise use defaults optimized for insights
        params = custom_params or {}

        # Set default parameters optimized for the stock's characteristics
        rsi_period = params.get("rsi_period", 14)
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)

        # Adjust parameters based on insights
        if insights["volatility"]["level"] == "high":
            rsi_overbought = params.get("rsi_overbought", 75)  # More conservative
            rsi_oversold = params.get("rsi_oversold", 25)      # More conservative

        # Generate strategy code
        strategy_code = f"""#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
#
# Momentum Strategy generated based on stock analysis
#
# This strategy uses RSI and MACD indicators to identify momentum
# and generate trading signals.

import backtrader as bt
import datetime


class MomentumStrategy(bt.Strategy):
    params = (
        ('rsi_period', {rsi_period}),
        ('rsi_overbought', {rsi_overbought}),
        ('rsi_oversold', {rsi_oversold}),
        ('macd1', 12),
        ('macd2', 26),
        ('macdsig', 9),
        ('target_annual_return', {risk_params.get('target_annual_return', 0.15)}),
        ('max_drawdown', {risk_params.get('max_drawdown', 0.20)}),
        ('trail_percent', {insights["volatility"]["atr_percentage"]}),
    )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{{dt.isoformat()}} {{txt}}')

    def __init__(self):
        # Keep track of pending orders
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add RSI indicator
        self.rsi = bt.indicators.RelativeStrengthIndex(
            period=self.params.rsi_period
        )

        # Add MACD indicator
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.params.macd1,
            period_me2=self.params.macd2,
            period_signal=self.params.macdsig
        )

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted - no action required
            return

        # Check if an order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {{order.executed.price:.2f}}, '
                         f'Cost: {{order.executed.value:.2f}}, '
                         f'Comm: {{order.executed.comm:.2f}}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(f'SELL EXECUTED, Price: {{order.executed.price:.2f}}, '
                         f'Cost: {{order.executed.value:.2f}}, '
                         f'Comm: {{order.executed.comm:.2f}}')

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Reset orders
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log(f'OPERATION PROFIT, GROSS: {{trade.pnl:.2f}}, NET: {{trade.pnlcomm:.2f}}')

    def next(self):
        # Check if an order is pending
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            # Not in the market, look for buy signal

            # Buy Signal: RSI below oversold threshold and MACD line crosses above signal line
            if self.rsi < self.params.rsi_oversold and self.macd.macd > self.macd.signal:
                self.log(f'BUY CREATE, {{self.data.close[0]:.2f}}')
                self.order = self.buy()

                # Set trailing stop loss based on volatility
                self.sell(exectype=bt.Order.StopTrail, trailpercent=self.params.trail_percent)
        else:
            # Already in the market, look for sell signal

            # Sell Signal: RSI above overbought threshold and MACD line crosses below signal line
            if self.rsi > self.params.rsi_overbought and self.macd.macd < self.macd.signal:
                self.log(f'SELL CREATE, {{self.data.close[0]:.2f}}')
                self.order = self.sell()


if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(MomentumStrategy)

    # Set up the data feed
    data = bt.feeds.YahooFinanceCSVData(
        dataname='data.csv',
        fromdate=datetime.datetime(2019, 1, 1),
        todate=datetime.datetime(2020, 12, 31),
        reverse=False)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)

    # Set the commission - 0.1% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=0.001)

    # Print out the starting conditions
    print(f'Starting Portfolio Value: {{cerebro.broker.getvalue():.2f}}')

    # Run over everything
    cerebro.run()

    # Print out the final result
    print(f'Final Portfolio Value: {{cerebro.broker.getvalue():.2f}}')

    # Plot the result
    cerebro.plot()
"""
        return strategy_code

    def _generate_trend_following_strategy(
        self,
        insights: Dict,
        risk_params: Dict,
        custom_params: Optional[Dict] = None
    ) -> str:
        """Generate a trend-following trading strategy."""
        # Use custom params if provided, otherwise use defaults optimized for insights
        params = custom_params or {}

        # Determine optimal parameters based on stock volatility and trend
        fast_period = params.get("fast_period", 20)
        slow_period = params.get("slow_period", 50)

        # Adjust parameters based on volatility
        if insights["volatility"]["level"] == "high":
            fast_period = params.get("fast_period", 25)  # Slower for high volatility
            slow_period = params.get("slow_period", 75)  # Longer trend for high volatility
        elif insights["volatility"]["level"] == "low":
            fast_period = params.get("fast_period", 15)  # Faster for low volatility
            slow_period = params.get("slow_period", 40)  # Shorter trend for low volatility

        # Generate strategy code
        strategy_code = f"""#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
#
# Trend Following Strategy generated based on stock analysis
#
# This strategy uses dual moving averages to follow trends
# and generate trading signals.

import backtrader as bt
import datetime


class TrendFollowingStrategy(bt.Strategy):
    params = (
        ('fast_period', {fast_period}),
        ('slow_period', {slow_period}),
        ('atr_period', 14),
        ('atr_distance', 2.0),
        ('target_annual_return', {risk_params.get('target_annual_return', 0.15)}),
        ('max_drawdown', {risk_params.get('max_drawdown', 0.20)}),
    )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{{dt.isoformat()}} {{txt}}')

    def __init__(self):
        # Keep track of pending orders
        self.order = None

        # Add moving average indicators
        self.fast_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.fast_period
        )
        self.slow_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.slow_period
        )

        # Add crossover indicator
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

        # Add ATR for stop loss calculation
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted - no action required
            return

        # Check if an order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {{order.executed.price:.2f}}, '
                         f'Cost: {{order.executed.value:.2f}}, '
                         f'Comm: {{order.executed.comm:.2f}}')
            else:
                self.log(f'SELL EXECUTED, Price: {{order.executed.price:.2f}}, '
                         f'Cost: {{order.executed.value:.2f}}, '
                         f'Comm: {{order.executed.comm:.2f}}')

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Reset orders
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log(f'OPERATION PROFIT, GROSS: {{trade.pnl:.2f}}, NET: {{trade.pnlcomm:.2f}}')

    def next(self):
        # Check if an order is pending
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            # Not in the market, look for buy signal

            # Buy when fast MA crosses above slow MA
            if self.crossover > 0:
                self.log(f'BUY CREATE, {{self.data.close[0]:.2f}}')
                self.order = self.buy()

                # Calculate stop loss based on ATR
                stop_price = self.data.close[0] - self.params.atr_distance * self.atr[0]
                self.sell(exectype=bt.Order.Stop, price=stop_price)

        else:
            # Already in the market, look for sell signal

            # Sell when fast MA crosses below slow MA
            if self.crossover < 0:
                self.log(f'SELL CREATE, {{self.data.close[0]:.2f}}')
                self.order = self.sell()


if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(TrendFollowingStrategy)

    # Set up the data feed
    data = bt.feeds.YahooFinanceCSVData(
        dataname='data.csv',
        fromdate=datetime.datetime(2019, 1, 1),
        todate=datetime.datetime(2020, 12, 31),
        reverse=False)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)

    # Set the commission - 0.1% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=0.001)

    # Print out the starting conditions
    print(f'Starting Portfolio Value: {{cerebro.broker.getvalue():.2f}}')

    # Run over everything
    cerebro.run()

    # Print out the final result
    print(f'Final Portfolio Value: {{cerebro.broker.getvalue():.2f}}')

    # Plot the result
    cerebro.plot()
"""
        return strategy_code

    def _generate_mean_reversion_strategy(self, insights: Dict, risk_params: Dict, custom_params: Optional[Dict] = None) -> str:
        """Generate placeholders for other strategy types."""
        # Template code (truncated for brevity)
        return """
# Mean Reversion Strategy
import backtrader as bt

class MeanReversionStrategy(bt.Strategy):
    # Implementation goes here
    pass
"""

    def _generate_dual_ma_crossover_strategy(self, insights: Dict, risk_params: Dict, custom_params: Optional[Dict] = None) -> str:
        return """
# Dual MA Crossover Strategy
import backtrader as bt

class DualMACrossoverStrategy(bt.Strategy):
    # Implementation goes here
    pass
"""

    def _generate_macd_strategy(self, insights: Dict, risk_params: Dict, custom_params: Optional[Dict] = None) -> str:
        return """
# MACD Strategy
import backtrader as bt

class MACDStrategy(bt.Strategy):
    # Implementation goes here
    pass
"""

    def _generate_rsi_strategy(self, insights: Dict, risk_params: Dict, custom_params: Optional[Dict] = None) -> str:
        return """
# RSI Strategy
import backtrader as bt

class RSIStrategy(bt.Strategy):
    # Implementation goes here
    pass
"""

    def _generate_bollinger_bands_strategy(self, insights: Dict, risk_params: Dict, custom_params: Optional[Dict] = None) -> str:
        return """
# Bollinger Bands Strategy
import backtrader as bt

class BollingerBandsStrategy(bt.Strategy):
    # Implementation goes here
    pass
"""

    def _generate_custom_strategy(self, insights: Dict, risk_params: Dict, custom_params: Optional[Dict] = None) -> str:
        return """
# Custom Combined Strategy
import backtrader as bt

class CustomStrategy(bt.Strategy):
    # Implementation goes here
    pass
"""

    def _momentum_strategy_template(self) -> str:
        return """
# Momentum Strategy Template
import backtrader as bt

class MomentumStrategy(bt.Strategy):
    params = (
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
        ('macd1', 12),
        ('macd2', 26),
        ('macdsig', 9),
    )

    def __init__(self):
        # Add your indicators here
        pass

    def next(self):
        # Add your trading logic here
        pass
"""

    def _trend_following_strategy_template(self) -> str:
        return """
# Trend Following Strategy Template
import backtrader as bt

class TrendFollowingStrategy(bt.Strategy):
    params = (
        ('fast_period', 20),
        ('slow_period', 50),
    )

    def __init__(self):
        # Add your indicators here
        pass

    def next(self):
        # Add your trading logic here
        pass
"""

    def _mean_reversion_strategy_template(self) -> str:
        return """
# Mean Reversion Strategy Template
import backtrader as bt

class MeanReversionStrategy(bt.Strategy):
    params = (
        ('bb_period', 20),
        ('bb_devfactor', 2),
    )

    def __init__(self):
        # Add your indicators here
        pass

    def next(self):
        # Add your trading logic here
        pass
"""

    def _dual_ma_crossover_strategy_template(self) -> str:
        return """
# Dual MA Crossover Strategy Template
import backtrader as bt

class DualMACrossoverStrategy(bt.Strategy):
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
    )

    def __init__(self):
        # Add your indicators here
        pass

    def next(self):
        # Add your trading logic here
        pass
"""

    def _macd_strategy_template(self) -> str:
        return """
# MACD Strategy Template
import backtrader as bt

class MACDStrategy(bt.Strategy):
    params = (
        ('macd1', 12),
        ('macd2', 26),
        ('macdsig', 9),
    )

    def __init__(self):
        # Add your indicators here
        pass

    def next(self):
        # Add your trading logic here
        pass
"""

    def _rsi_strategy_template(self) -> str:
        return """
# RSI Strategy Template
import backtrader as bt

class RSIStrategy(bt.Strategy):
    params = (
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
    )

    def __init__(self):
        # Add your indicators here
        pass

    def next(self):
        # Add your trading logic here
        pass
"""

    def _bollinger_bands_strategy_template(self) -> str:
        return """
# Bollinger Bands Strategy Template
import backtrader as bt

class BollingerBandsStrategy(bt.Strategy):
    params = (
        ('bb_period', 20),
        ('bb_devfactor', 2),
    )

    def __init__(self):
        # Add your indicators here
        pass

    def next(self):
        # Add your trading logic here
        pass
"""

    def _custom_strategy_template(self) -> str:
        return """
# Custom Strategy Template
import backtrader as bt

class CustomStrategy(bt.Strategy):
    params = (
        # Define your parameters here
    )

    def __init__(self):
        # Add your indicators here
        pass

    def next(self):
        # Add your trading logic here
        pass
"""

    def _generate_dynamic_strategy(
        self,
        insights: Dict,
        risk_params: Dict,
        custom_params: Optional[Dict] = None,
        analysis_results: Optional[Dict] = None
    ) -> str:
        """
        根据市场分析结果动态生成交易策略。
        这是一个高级策略生成方法，能够根据市场条件和技术指标灵活组合策略逻辑。

        参数:
            insights: 市场分析洞察
            risk_params: 风险参数
            custom_params: 自定义参数
            analysis_results: 完整分析结果

        返回:
            生成的策略代码
        """
        # 设置默认值
        custom_params = custom_params or {}

        # 决定使用哪些指标
        use_sma = custom_params.get("use_sma", True)
        use_rsi = custom_params.get("use_rsi", True)
        use_macd = custom_params.get("use_macd", True)
        use_bbands = custom_params.get("use_bbands", True)
        use_atr = custom_params.get("use_atr", True)

        # 获取指标周期
        fast_period = custom_params.get("fast_period", 10)
        slow_period = custom_params.get("slow_period", 30)
        rsi_period = custom_params.get("rsi_period", 14)
        rsi_overbought = custom_params.get("rsi_overbought", 70)
        rsi_oversold = custom_params.get("rsi_oversold", 30)
        macd_fast = custom_params.get("macd_fast", 12)
        macd_slow = custom_params.get("macd_slow", 26)
        macd_signal = custom_params.get("macd_signal", 9)
        bb_period = custom_params.get("bb_period", 20)
        bb_dev = custom_params.get("bb_dev", 2.0)
        atr_period = custom_params.get("atr_period", 14)
        atr_multiplier = custom_params.get("atr_multiplier", 2.0)

        # 基于市场分析动态调整参数
        market_trend = insights.get("trend", {}).get("direction", "neutral")
        market_volatility = insights.get("volatility", {}).get("level", "medium")

        # 根据市场趋势和波动性构建入场条件
        strategy_logic = ""
        entry_conditions = []
        exit_conditions = []

        # 构建指标定义部分
        indicators = []

        if use_sma:
            indicators.append(f"        # 移动平均线\n        self.fast_sma = bt.indicators.SimpleMovingAverage(self.data.close, period={fast_period})\n        self.slow_sma = bt.indicators.SimpleMovingAverage(self.data.close, period={slow_period})")

        if use_rsi:
            indicators.append(f"        # 相对强弱指标\n        self.rsi = bt.indicators.RelativeStrengthIndex(period={rsi_period})")

        if use_macd:
            indicators.append(f"        # MACD指标\n        self.macd = bt.indicators.MACD(self.data.close, period_me1={macd_fast}, period_me2={macd_slow}, period_signal={macd_signal})")

        if use_bbands:
            indicators.append(f"        # 布林带\n        self.bbands = bt.indicators.BollingerBands(self.data.close, period={bb_period}, devfactor={bb_dev})")

        if use_atr:
            indicators.append(f"        # 平均真实波幅\n        self.atr = bt.indicators.ATR(self.data, period={atr_period})")

        # 根据市场趋势和波动性构建入场条件
        if market_trend == "uptrend" or market_trend == "strong_uptrend":
            # 上升趋势策略
            if use_sma:
                entry_conditions.append("self.fast_sma > self.slow_sma")
            if use_rsi:
                entry_conditions.append(f"self.rsi > 50")  # 在上升趋势中使用更高的RSI门槛
            if use_macd:
                entry_conditions.append("self.macd.macd > self.macd.signal")
            if use_bbands:
                if market_volatility == "high":
                    entry_conditions.append("self.data.close > self.bbands.mid")  # 高波动性下更保守
                else:
                    entry_conditions.append("self.data.close > self.bbands.top")  # 低波动性下更激进

        elif market_trend == "downtrend" or market_trend == "strong_downtrend":
            # 下降趋势策略 - 做空或寻找反转点
            if use_sma:
                entry_conditions.append("self.fast_sma < self.slow_sma")  # 可能的做空信号
            if use_rsi:
                entry_conditions.append(f"self.rsi < {rsi_oversold}")  # 超卖可能反弹
            if use_macd:
                entry_conditions.append("self.macd.macd < self.macd.signal")
            if use_bbands:
                entry_conditions.append("self.data.close < self.bbands.bot")  # 突破下轨

        else:  # 中性或不确定趋势
            # 组合策略
            if use_sma and use_rsi:
                entry_conditions.append(f"(self.fast_sma > self.slow_sma and self.rsi > 50) or (self.rsi < {rsi_oversold})")
            elif use_macd and use_bbands:
                entry_conditions.append("(self.macd.macd > self.macd.signal and self.data.close > self.bbands.mid) or (self.data.close < self.bbands.bot)")
            else:
                # 默认条件
                if use_sma:
                    entry_conditions.append("self.fast_sma > self.slow_sma")
                if use_rsi:
                    entry_conditions.append(f"self.rsi < {rsi_oversold} or self.rsi > {rsi_overbought}")

        # 构建出场条件
        if market_volatility == "high":
            # 高波动性市场中更积极的止损和止盈
            if use_atr:
                exit_conditions.append("abs(self.data.close[0] - self.position.price) > self.atr[0] * 2")
            if use_rsi:
                exit_conditions.append(f"(self.position.size > 0 and self.rsi > {rsi_overbought + 10}) or (self.position.size < 0 and self.rsi < {rsi_oversold - 10})")
        else:
            # 正常波动性市场
            if use_sma:
                exit_conditions.append("(self.position.size > 0 and self.fast_sma < self.slow_sma) or (self.position.size < 0 and self.fast_sma > self.slow_sma)")
            if use_rsi:
                exit_conditions.append(f"(self.position.size > 0 and self.rsi > {rsi_overbought}) or (self.position.size < 0 and self.rsi < {rsi_oversold})")
            if use_macd:
                exit_conditions.append("(self.position.size > 0 and self.macd.macd < self.macd.signal) or (self.position.size < 0 and self.macd.macd > self.macd.signal)")

        # 添加默认止损
        stop_loss_pct = risk_params.get("max_drawdown", 0.2) / 1.5  # 使用最大回撤参数计算止损百分比
        exit_conditions.append(f"(self.position.size > 0 and self.data.close[0] < self.position.price * (1.0 - {stop_loss_pct:.4f})) or (self.position.size < 0 and self.data.close[0] > self.position.price * (1.0 + {stop_loss_pct:.4f}))")

        # 构建策略
        strategy_template = f"""#!/usr/bin/env python
# -*- coding: utf-8 -*-

import backtrader as bt
import numpy as np
import pandas as pd
from datetime import datetime

class DynamicStrategy(bt.Strategy):


    params = (
        ('fast_period', {fast_period}),
        ('slow_period', {slow_period}),
        ('rsi_period', {rsi_period}),
        ('rsi_overbought', {rsi_overbought}),
        ('rsi_oversold', {rsi_oversold}),
        ('macd_fast', {macd_fast}),
        ('macd_slow', {macd_slow}),
        ('macd_signal', {macd_signal}),
        ('bb_period', {bb_period}),
        ('bb_dev', {bb_dev}),
        ('atr_period', {atr_period}),
        ('atr_multiplier', {atr_multiplier}),
        ('stop_loss_pct', {stop_loss_pct:.4f}),
        ('target_profit_pct', {risk_params.get("target_annual_return", 0.15) / 4:.4f}),
    )

    def __init__(self):
        \"\"\"初始化策略，定义技术指标\"\"\"
{chr(10).join(indicators)}

        # 策略洞察
        \"\"\"
        市场分析洞察:
        - 趋势: {insights.get('trend', {}).get('direction', 'neutral')} (强度: {insights.get('trend', {}).get('strength', 'weak')})
        - 动量: {insights.get('momentum', {}).get('state', 'neutral')} (强度: {insights.get('momentum', {}).get('strength', 'weak')})
        - 波动性: {insights.get('volatility', {}).get('level', 'medium')} (ATR%: {insights.get('volatility', {}).get('atr_percentage', 2.0)})
        - RSI水平: {insights.get('indicators', {}).get('rsi_level', 50)}
        - MACD信号: {insights.get('indicators', {}).get('macd_signal', 'neutral')}
        - 布林带位置: {insights.get('indicators', {}).get('bollinger_position', 'middle')}
        \"\"\"

        # 交易统计
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.bar_executed = None

        # 交易计数器
        self.trade_count = 0
        self.profitable_trades = 0

    def log(self, txt, dt=None):
        \"\"\"日志函数，用于输出消息\"\"\"
        dt = dt or self.datas[0].datetime.date(0)
        print(f'%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        \"\"\"订单状态通知\"\"\"
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行: 价格: {order.executed.price:.2f}, 成本: {order.executed.value:.2f}, 手续费: {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                profit = (order.executed.price - self.buyprice) * order.executed.size if self.buyprice else 0
                self.log(f'卖出执行: 价格: {order.executed.price:.2f}, 利润: {profit:.2f}')

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/保证金不足/拒绝')

        self.order = None

    def notify_trade(self, trade):
        \"\"\"交易结束通知\"\"\"
        if not trade.isclosed:
            return

        self.trade_count += 1
        if trade.pnl > 0:
            self.profitable_trades += 1

        self.log(f'交易利润: 毛利={{trade.pnl:.2f}}, 净利={{trade.pnlcomm:.2f}}')

    def next(self):
        \"\"\"主策略逻辑 - 每个交易周期执行一次\"\"\"
        self.log(f'收盘价: {{self.data.close[0]:.2f}}')

        # 如果有未完成的订单，不执行新订单
        if self.order:
            return

        # 如果没有持仓
        if not self.position:
            # 入场条件
            if {" and ".join(entry_conditions)}:
                self.log('生成买入信号')

                # 计算头寸规模
                cash = self.broker.getcash()
                price = self.data.close[0]
                size = int(cash * 0.95 / price)  # 使用95%的可用资金

                if size > 0:
                    self.log(f'买入: {{size}}股')
                    self.order = self.buy(size=size)

        # 如果有持仓
        else:
            # 出场条件
            if {" or ".join(exit_conditions)}:
                self.log('生成卖出信号')
                self.order = self.close()  # 平仓当前持仓

    def stop(self):
        \"\"\"策略结束时执行\"\"\"
        win_rate = self.profitable_trades / self.trade_count * 100 if self.trade_count > 0 else 0
        self.log(f'交易总数: {{self.trade_count}}, 获利交易: {{self.profitable_trades}}, 胜率: {{win_rate:.2f}}%')
        self.log(f'起始资金: {{self.broker.startingcash:.2f}}, 最终资金: {{self.broker.getvalue():.2f}}, 总收益率: {{((self.broker.getvalue() / self.broker.startingcash) - 1) * 100:.2f}}%')

    # 用于回测的主程序
    if __name__ == '__main__':
         # 此部分代码用于直接运行策略进行回测
        print("这是一个自动生成的策略模板，请与回测框架结合使用")
"""
        return strategy_template

# For testing
if __name__ == "__main__":
    import asyncio
    import json

    async def test():
        try:
            tool = StrategyGeneratorTool()

            # Test listing available strategies
            result = await tool.execute(command="list_available_strategies")
            print("Available Strategies:")
            print(result.output)

            # Test getting a strategy template
            result = await tool.execute(command="get_strategy_template", strategy_type="momentum")
            print("\nMomentum Strategy Template:")
            print(result.output)

            # Test generating a strategy
            mock_analysis = {
                "technical_analysis": {
                    "trend": {"is_uptrend": True},
                    "momentum": {"rsi": 45, "is_macd_bullish": True},
                    "volatility": {"volatility_percentage": 2.5},
                    "bollinger_bands": {"price_relative_to_band": "MIDDLE"}
                },
                "market_condition": {
                    "overall_assessment": {"condition": "FAVORABLE"}
                }
            }

            mock_risk_params = {
                "target_annual_return": 0.20,
                "max_drawdown": 0.15,
                "sharpe_ratio": 1.2,
                "profit_loss_ratio": 1.8
            }

            result = await tool.execute(
                command="generate_strategy",
                strategy_type="momentum",
                analysis_results=mock_analysis,
                risk_params=mock_risk_params
            )

            print("\nGenerated Strategy:")
            print(result.output)

        except Exception as e:
            print(f"Error in test: {str(e)}")

    asyncio.run(test())
