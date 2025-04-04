#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试策略生成器工具
"""

import asyncio

from app.tool.strategy_generator import StrategyGeneratorTool


async def test():
    try:
        tool = StrategyGeneratorTool()

        # 测试生成策略
        mock_analysis = {
            "technical_analysis": {
                "trend": {"is_uptrend": True},
                "momentum": {"rsi": 45, "is_macd_bullish": True},
                "volatility": {"volatility_percentage": 2.5},
                "bollinger_bands": {"price_relative_to_band": "MIDDLE"},
            },
            "market_condition": {"overall_assessment": {"condition": "FAVORABLE"}},
        }

        mock_risk_params = {
            "target_annual_return": 0.20,
            "max_drawdown": 0.15,
            "sharpe_ratio": 1.2,
            "profit_loss_ratio": 1.8,
        }

        result = await tool.execute(
            command="generate_strategy",
            analysis_results=mock_analysis,
            risk_params=mock_risk_params,
        )

        if result.error:
            print(f"测试失败: {result.error}")
        else:
            print("测试成功!")
            print("\n生成的策略摘要:")
            print(result.output[:200] + "..." if result.output else "无输出")

    except Exception as e:
        print(f"测试出错: {str(e)}")


if __name__ == "__main__":
    asyncio.run(test())
