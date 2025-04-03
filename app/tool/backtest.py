"""Backtest tool for evaluating trading strategies."""

import json
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Any
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import backtrader as bt

from app.tool.base import BaseTool, ToolResult, save_to_workspace, generate_timestamp_id, ensure_workspace_dir
from app.logger import logger


class BacktestAnalyzer(bt.Analyzer):
    """Custom analyzer to track performance metrics during backtest."""

    def get_analysis(self):
        return self.strategy.stats

    def start(self):
        self.strategy.stats = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'even': 0,
            'win_streak': 0,
            'loss_streak': 0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'total_net_profit': 0.0,
            'avg_profit_per_trade': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'annual_return_pct': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'profit_loss_ratio': 0.0,
            'win_rate': 0.0,
            'current_streak': 0
        }

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.strategy.stats['total_trades'] += 1

        profit = trade.pnlcomm
        self.strategy.stats['total_net_profit'] += profit

        if profit > 0:
            self.strategy.stats['wins'] += 1
            self.strategy.stats['current_streak'] = max(0, self.strategy.stats['current_streak'] + 1)
            self.strategy.stats['win_streak'] = max(self.strategy.stats['win_streak'], self.strategy.stats['current_streak'])
            self.strategy.stats['largest_win'] = max(self.strategy.stats['largest_win'], profit)
        elif profit < 0:
            self.strategy.stats['losses'] += 1
            self.strategy.stats['current_streak'] = min(0, self.strategy.stats['current_streak'] - 1)
            self.strategy.stats['loss_streak'] = min(self.strategy.stats['loss_streak'], self.strategy.stats['current_streak'])
            self.strategy.stats['largest_loss'] = min(self.strategy.stats['largest_loss'], profit)
        else:
            self.strategy.stats['even'] += 1

    def stop(self):
        stats = self.strategy.stats

        # Calculate win rate
        if stats['total_trades'] > 0:
            stats['win_rate'] = stats['wins'] / stats['total_trades'] * 100.0
            stats['avg_profit_per_trade'] = stats['total_net_profit'] / stats['total_trades']

        # Calculate average win and loss
        if stats['wins'] > 0:
            stats['avg_win'] = stats['largest_win'] / stats['wins']
        if stats['losses'] > 0:
            stats['avg_loss'] = stats['largest_loss'] / stats['losses']

        # Calculate profit factor
        if stats['avg_loss'] != 0:
            stats['profit_factor'] = abs(stats['avg_win'] / stats['avg_loss'])
            stats['profit_loss_ratio'] = abs(stats['avg_win'] / stats['avg_loss'])

        # Max drawdown and other metrics calculated by standard analyzers
        # will be incorporated later


class BacktestTool(BaseTool):
    """Tool for backtesting trading strategies using the backtrader framework."""

    name: str = "backtest"
    description: str = (
        "Performs backtesting on trading strategies using historical stock data. "
        "Evaluates strategy performance, calculates key metrics like returns, drawdowns, "
        "and Sharpe ratio, and generates performance reports."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Backtest command to execute",
                "enum": [
                    "run_backtest",
                    "analyze_results",
                    "plot_performance",
                ],
            },
            "strategy_file": {
                "type": "string",
                "description": "Path to the strategy file to backtest",
            },
            "data_file": {
                "type": "string",
                "description": "Path to the data file for backtesting",
            },
            "start_date": {
                "type": "string",
                "description": "Start date for backtest in YYYY-MM-DD format",
            },
            "end_date": {
                "type": "string",
                "description": "End date for backtest in YYYY-MM-DD format",
            },
            "initial_capital": {
                "type": "number",
                "description": "Initial capital for the backtest",
            },
            "commission": {
                "type": "number",
                "description": "Commission rate for trades (0.001 = 0.1%)",
            },
            "strategy_params": {
                "type": "object",
                "description": "Parameters to pass to the strategy",
            },
            "result_id": {
                "type": "string",
                "description": "ID of a previous backtest result to analyze or plot",
            },
        },
        "required": ["command"],
    }

    async def execute(
        self,
        command: str,
        strategy_file: Optional[str] = None,
        data_file: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_capital: Optional[float] = 100000.0,
        commission: Optional[float] = 0.001,
        strategy_params: Optional[Dict] = None,
        result_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Execute the backtest command."""

        try:
            if command == "run_backtest":
                if not strategy_file or not data_file:
                    return ToolResult(error="策略文件和数据文件是运行回测的必要参数")

                # Convert dates if provided
                start_date_obj = None
                end_date_obj = None

                if start_date:
                    try:
                        start_date_obj = datetime.datetime.strptime(start_date, "%Y-%m-%d")
                    except ValueError:
                        return ToolResult(error=f"无效的开始日期格式: {start_date}，请使用YYYY-MM-DD格式")

                if end_date:
                    try:
                        end_date_obj = datetime.datetime.strptime(end_date, "%Y-%m-%d")
                    except ValueError:
                        return ToolResult(error=f"无效的结束日期格式: {end_date}，请使用YYYY-MM-DD格式")

                # Run the backtest
                result = self._run_backtest(
                    strategy_file=strategy_file,
                    data_file=data_file,
                    start_date=start_date_obj,
                    end_date=end_date_obj,
                    initial_capital=initial_capital,
                    commission=commission,
                    strategy_params=strategy_params or {}
                )

                return result

            elif command == "analyze_results":
                if not result_id:
                    return ToolResult(error="分析结果需要提供回测结果ID")

                result = self._analyze_results(result_id=result_id)
                return result

            elif command == "plot_performance":
                if not result_id:
                    return ToolResult(error="绘制性能图表需要提供回测结果ID")

                result = self._plot_performance(result_id=result_id)
                return result

            else:
                return ToolResult(error=f"未知命令: {command}")

        except Exception as e:
            return ToolResult(error=f"执行命令 {command} 时出错: {str(e)}")

    def _run_backtest(
        self,
        strategy_file: str,
        data_file: str,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        strategy_params: Dict = {}
    ) -> ToolResult:
        """
        使用提供的策略和数据运行回测。
        回测结果将保存到工作空间。

        参数:
            strategy_file: 策略文件路径
            data_file: 数据文件路径
            start_date: 开始日期
            end_date: 结束日期
            initial_capital: 初始资金
            commission: 交易佣金率
            strategy_params: 策略参数

        返回:
            包含回测结果信息的ToolResult对象
        """
        # 验证输入文件
        if not os.path.exists(strategy_file):
            return ToolResult(error=f"策略文件不存在: {strategy_file}")

        if not os.path.exists(data_file):
            return ToolResult(error=f"数据文件不存在: {data_file}")

        # 动态导入策略
        import importlib.util
        import sys

        try:
            # 创建唯一模块名
            module_name = f"strategy_module_{os.path.basename(strategy_file).replace('.py', '')}"

            # 加载模块
            spec = importlib.util.spec_from_file_location(module_name, strategy_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # 在模块中查找Strategy类
            strategy_class = None
            for name in dir(module):
                obj = getattr(module, name)
                if isinstance(obj, type) and issubclass(obj, bt.Strategy) and obj != bt.Strategy:
                    strategy_class = obj
                    break

            if not strategy_class:
                return ToolResult(error="在策略文件中未找到有效的策略类")

        except Exception as e:
            return ToolResult(error=f"导入策略文件时出错: {str(e)}")

        # 加载数据
        try:
            # 根据文件扩展名确定数据格式
            if data_file.endswith('.csv'):
                # 对于CSV文件，假设文件具有列：datetime, open, high, low, close, volume
                data = bt.feeds.GenericCSVData(
                    dataname=data_file,
                    dtformat='%Y-%m-%d',
                    datetime=0,
                    open=1,
                    high=2,
                    low=3,
                    close=4,
                    volume=5,
                    openinterest=-1,
                    fromdate=start_date,
                    todate=end_date
                )
            else:
                return ToolResult(error=f"不支持的数据文件格式: {data_file}")

        except Exception as e:
            return ToolResult(error=f"加载数据文件时出错: {str(e)}")

        # 创建和运行回测
        try:
            # 创建cerebro实例
            cerebro = bt.Cerebro()

            # 添加策略
            cerebro.addstrategy(strategy_class, **strategy_params)

            # 添加数据
            cerebro.adddata(data)

            # 设置初始资金
            cerebro.broker.setcash(initial_capital)

            # 设置佣金
            cerebro.broker.setcommission(commission=commission)

            # 添加分析器
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            cerebro.addanalyzer(BacktestAnalyzer, _name='custom')

            # 运行回测
            results = cerebro.run()

            # 提取结果
            strat = results[0]

            # 从分析器获取结果
            sharpe = strat.analyzers.sharpe.get_analysis()
            drawdown = strat.analyzers.drawdown.get_analysis()
            returns = strat.analyzers.returns.get_analysis()
            trades = strat.analyzers.trades.get_analysis()
            custom = strat.analyzers.custom.get_analysis()

            # 计算关键指标
            total_return = cerebro.broker.getvalue() / initial_capital - 1.0

            # 生成唯一回测ID
            backtest_id = generate_timestamp_id("backtest")

            # 准备结果对象
            results_obj = {
                "id": backtest_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "strategy": os.path.basename(strategy_file),
                "data": os.path.basename(data_file),
                "parameters": {
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                    "initial_capital": initial_capital,
                    "commission": commission,
                    "strategy_params": strategy_params
                },
                "summary": {
                    "final_value": cerebro.broker.getvalue(),
                    "total_return": total_return,
                    "total_return_pct": total_return * 100.0,
                    "annual_return": returns.get('ravg', 0.0) * 100.0 if hasattr(returns, 'get') else 0.0,
                    "sharpe_ratio": sharpe.get('sharperatio', 0.0) if hasattr(sharpe, 'get') else 0.0,
                    "max_drawdown": drawdown.get('max', {}).get('drawdown', 0.0) * 100.0 if hasattr(drawdown, 'get') else 0.0,
                    "max_drawdown_length": drawdown.get('max', {}).get('len', 0) if hasattr(drawdown, 'get') else 0,
                    "total_trades": trades.get('total', {}).get('total', 0) if hasattr(trades, 'get') else 0,
                    "win_rate": custom.get('win_rate', 0.0) if custom else 0.0,
                    "profit_loss_ratio": custom.get('profit_loss_ratio', 0.0) if custom else 0.0
                },
                "detailed": {
                    "trades": trades if hasattr(trades, 'get') else {},
                    "custom": custom if custom else {},
                    "equity_curve": [cerebro.broker.get_value()]  # 在未来的增强中我们将添加适当的权益曲线
                }
            }

            # 保存结果到工作空间
            results_file = f"{backtest_id}_results.json"
            results_file_path = save_to_workspace(results_obj, results_file, "backtest_results", is_json=True)
            logger.info(f"回测结果已保存到: {results_file_path}")

            # 生成图表并保存
            plot_file = f"{backtest_id}_plot.png"
            plot_path = os.path.join(ensure_workspace_dir("backtest_results"), plot_file)

            try:
                # 创建图表
                fig = cerebro.plot(style='candlestick', barup='red', bardown='green',
                                  volup='red', voldown='green', grid=True,
                                  returnfig=True)

                # 保存图表
                if fig and len(fig) > 0 and len(fig[0]) > 0:
                    fig[0][0].savefig(plot_path)
                    logger.info(f"回测图表已保存到: {plot_path}")
            except Exception as plot_error:
                logger.error(f"生成图表时出错: {str(plot_error)}")

            # 返回结果
            return ToolResult(output=json.dumps({
                "backtest_id": backtest_id,
                "strategy": os.path.basename(strategy_file),
                "data": os.path.basename(data_file),
                "summary": results_obj["summary"],
                "results_file": results_file_path,
                "plot_file": plot_path if os.path.exists(plot_path) else None,
                "message": f"回测完成。结果已保存到 {results_file_path}"
            }, ensure_ascii=False, indent=2))

        except Exception as e:
            logger.exception(f"执行回测时出错: {str(e)}")
            return ToolResult(error=f"执行回测时出错: {str(e)}")

    def _analyze_results(self, result_id: str) -> ToolResult:
        """Analyze the results of a backtest."""

        # Look for the results file
        results_dir = os.path.join(os.getcwd(), "workspace", "backtest_results")
        results_file = os.path.join(results_dir, f"{result_id}.json")

        if not os.path.exists(results_file):
            return ToolResult(error=f"未找到回测结果文件: {results_file}")

        # Load the results
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)

            # Extract and format the analysis
            summary = results["summary"]

            # Enhanced analysis - more detailed breakdown
            trades_analysis = results.get("detailed", {}).get("trades", {})
            custom_analysis = results.get("detailed", {}).get("custom", {})

            # Calculate more metrics if possible
            win_rate = summary.get("win_rate", 0.0)
            profit_loss_ratio = summary.get("profit_loss_ratio", 0.0)

            # Recovery factor = total return / max drawdown
            recovery_factor = 0.0
            if summary.get("max_drawdown", 0.0) > 0:
                recovery_factor = summary.get("total_return_pct", 0.0) / summary.get("max_drawdown", 1.0)

            # Overall assessment
            assessment = {
                "strengths": [],
                "weaknesses": [],
                "recommendations": []
            }

            # Evaluate various aspects
            # Return assessment
            if summary.get("annual_return", 0.0) > 10.0:
                assessment["strengths"].append("策略年化收益率表现优异，超过10%")
            elif summary.get("annual_return", 0.0) < 5.0:
                assessment["weaknesses"].append("策略年化收益率较低，低于5%")
                assessment["recommendations"].append("考虑优化策略参数以提高收益率")

            # Risk assessment
            if summary.get("max_drawdown", 100.0) < 10.0:
                assessment["strengths"].append("策略最大回撤控制良好，低于10%")
            elif summary.get("max_drawdown", 0.0) > 20.0:
                assessment["weaknesses"].append("策略最大回撤较大，超过20%")
                assessment["recommendations"].append("增强风险管理，添加止损策略")

            # Sharpe ratio assessment
            if summary.get("sharpe_ratio", 0.0) > 1.0:
                assessment["strengths"].append("夏普比率良好，风险调整后收益具有竞争力")
            elif summary.get("sharpe_ratio", 0.0) < 0.5:
                assessment["weaknesses"].append("夏普比率较低，风险调整后收益不佳")
                assessment["recommendations"].append("减少交易频率或改进信号质量")

            # Trade quality assessment
            if win_rate > 60.0:
                assessment["strengths"].append(f"交易胜率高达{win_rate:.1f}%")
            elif win_rate < 40.0:
                assessment["weaknesses"].append(f"交易胜率较低，仅有{win_rate:.1f}%")
                assessment["recommendations"].append("改进入场信号质量，或优化出场时机")

            if profit_loss_ratio > 2.0:
                assessment["strengths"].append(f"盈亏比优秀: {profit_loss_ratio:.2f}")
            elif profit_loss_ratio < 1.0:
                assessment["weaknesses"].append(f"盈亏比不佳: {profit_loss_ratio:.2f}")
                assessment["recommendations"].append("延长盈利交易持有时间，缩短亏损交易持有时间")

            # Trading frequency assessment
            total_trades = summary.get("total_trades", 0)
            avg_trade_duration = custom_analysis.get("avg_trade_duration", 0)

            # Prepare the analysis results
            analysis = {
                "result_id": result_id,
                "strategy": results["strategy"],
                "data": results["data"],
                "parameters": results["parameters"],
                "performance_metrics": {
                    "returns": {
                        "total_return_pct": summary.get("total_return_pct", 0.0),
                        "annual_return_pct": summary.get("annual_return", 0.0),
                        "final_value": summary.get("final_value", 0.0),
                        "initial_capital": results["parameters"]["initial_capital"]
                    },
                    "risk": {
                        "max_drawdown_pct": summary.get("max_drawdown", 0.0),
                        "max_drawdown_duration": summary.get("max_drawdown_length", 0),
                        "sharpe_ratio": summary.get("sharpe_ratio", 0.0),
                        "recovery_factor": recovery_factor
                    },
                    "trade_statistics": {
                        "total_trades": total_trades,
                        "win_rate": win_rate,
                        "profit_loss_ratio": profit_loss_ratio,
                        "avg_profit_per_trade": custom_analysis.get("avg_profit_per_trade", 0.0),
                        "largest_win": custom_analysis.get("largest_win", 0.0),
                        "largest_loss": custom_analysis.get("largest_loss", 0.0),
                        "avg_win": custom_analysis.get("avg_win", 0.0),
                        "avg_loss": custom_analysis.get("avg_loss", 0.0),
                        "win_streak": custom_analysis.get("win_streak", 0),
                        "loss_streak": custom_analysis.get("loss_streak", 0)
                    }
                },
                "assessment": assessment
            }

            return ToolResult(output=json.dumps(analysis, ensure_ascii=False, indent=2))

        except Exception as e:
            return ToolResult(error=f"分析回测结果时出错: {str(e)}")

    def _plot_performance(self, result_id: str) -> ToolResult:
        """Plot the performance of a backtest and save the charts."""

        # Look for the results file
        results_dir = os.path.join(os.getcwd(), "workspace", "backtest_results")
        results_file = os.path.join(results_dir, f"{result_id}.json")

        if not os.path.exists(results_file):
            return ToolResult(error=f"未找到回测结果文件: {results_file}")

        # Check if a plot already exists
        plot_file = os.path.join(results_dir, f"{result_id}.png")
        if os.path.exists(plot_file):
            return ToolResult(output=json.dumps({
                "result_id": result_id,
                "plot_file": plot_file,
                "message": f"绩效图表已存在于 {plot_file}"
            }, ensure_ascii=False, indent=2))

        # Load the results
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)

            # Create simple performance charts
            # In a real-world scenario, we would extract the equity curve and other metrics
            # from the backtest results to create more detailed charts

            # For now, we'll just return a message about the existing plot
            return ToolResult(output=json.dumps({
                "result_id": result_id,
                "message": "绩效图表生成功能将在未来版本中完善。目前请使用backtrader的内置绘图功能。"
            }, ensure_ascii=False, indent=2))

        except Exception as e:
            return ToolResult(error=f"绘制绩效图表时出错: {str(e)}")


# For testing
if __name__ == "__main__":
    import asyncio

    async def test():
        tool = BacktestTool()

        # Example strategy file and data file for testing
        strategy_file = "./momentum_strategy.py"
        data_file = "./data.csv"

        # Run a backtest
        result = await tool.execute(
            command="run_backtest",
            strategy_file=strategy_file,
            data_file=data_file,
            start_date="2020-01-01",
            end_date="2021-01-01",
            initial_capital=100000.0,
            commission=0.001
        )

        print(result.output or result.error)

        if result.output:
            # Extract the result_id
            result_data = json.loads(result.output)
            result_id = result_data.get("backtest_id")

            if result_id:
                # Analyze the results
                analysis = await tool.execute(
                    command="analyze_results",
                    result_id=result_id
                )

                print(analysis.output or analysis.error)

                # Plot the performance
                plot = await tool.execute(
                    command="plot_performance",
                    result_id=result_id
                )

                print(plot.output or plot.error)

    asyncio.run(test())
