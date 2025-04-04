"""Strategy generator tool for creating trading strategies based on stock analysis."""

import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import re

from app.tool.base import BaseTool, ToolResult, save_to_workspace, generate_timestamp_id, ensure_workspace_dir
from app.logger import logger


class StrategyGeneratorTool(BaseTool):
    """A tool for generating backtrader trading strategies based on stock analysis."""

    name: str = "strategy_generator"
    description: str = (
        "生成基于股票分析结果的动态交易策略。"
        "根据技术指标、市场趋势和波动性自动创建适应性强的交易策略代码。"
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "策略生成命令",
                "enum": [
                    "generate_strategy",
                ],
            },
            "analysis_results": {
                "type": "object",
                "description": "股票分析结果，用于生成策略",
            },
            "risk_params": {
                "type": "object",
                "description": "策略风险参数",
                "properties": {
                    "target_annual_return": {"type": "number"},
                    "max_drawdown": {"type": "number"},
                    "sharpe_ratio": {"type": "number"},
                    "profit_loss_ratio": {"type": "number"},
                },
            },
            "custom_params": {
                "type": "object",
                "description": "策略自定义参数",
                "properties": {
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
        "required": ["command", "analysis_results"],
    }

    async def execute(
        self,
        command: str,
        analysis_results: Optional[Dict] = None,
        risk_params: Optional[Dict] = None,
        custom_params: Optional[Dict] = None,
        strategy_type: Optional[str] = None,  # 兼容旧接口，但不使用此参数
        **kwargs,
    ) -> ToolResult:
        """执行策略生成命令。"""

        try:
            if command == "generate_strategy":
                if not analysis_results:
                    return ToolResult(error="生成策略需要提供分析结果(analysis_results)")

                result = self._generate_strategy(analysis_results, risk_params, custom_params)
                return result
            else:
                return ToolResult(error=f"未知命令: {command}")

        except Exception as e:
            return ToolResult(error=f"执行命令 {command} 时出错: {str(e)}")

    def _generate_strategy(
        self,
        analysis_results: Dict,
        risk_params: Optional[Dict] = None,
        custom_params: Optional[Dict] = None
    ) -> ToolResult:
        """
        根据分析结果和参数生成动态交易策略。
        生成的策略代码将保存到工作空间。

        参数:
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

        # 生成动态策略代码
        strategy_code = self._generate_dynamic_strategy(strategy_insights, risk_params, custom_params, analysis_results)

        # 生成唯一ID用于策略文件
        strategy_id = generate_timestamp_id("dynamic_strategy")
        strategy_filename = f"{strategy_id}.py"

        # 保存策略到工作空间
        try:
            logger.info(f"尝试保存策略文件到: strategies")
            logger.info(f"策略代码长度: {len(strategy_code)}")
            strategy_file_path = save_to_workspace(strategy_code, strategy_filename, "strategies")

            # 保存策略元数据
            strategy_metadata = {
                "id": strategy_id,
                "strategy_type": "dynamic",
                "strategy_file": strategy_file_path,
                "risk_parameters": risk_params,
                "strategy_insights": strategy_insights,
                "custom_parameters": custom_params,
                "timestamp": datetime.now().isoformat(),
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
            "strategy_type": "dynamic",
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
        self.order = []
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
                self.log(f'买入执行: 价格: order.executed.price:.2f, 成本: order.executed.value:.2f, 手续费: order.executed.comm:.2f')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                profit = (order.executed.price - self.buyprice) * order.executed.size if self.buyprice else 0
                self.log(f'卖出执行: 价格: order.executed.price:.2f, 利润: profit:.2f')

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
"""
        return strategy_template




