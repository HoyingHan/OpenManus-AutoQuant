"""Strategy optimizer tool for optimizing trading strategies parameters."""

import json
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import numpy as np
import backtrader as bt
import itertools
import concurrent.futures

from app.tool.base import BaseTool, ToolResult, save_to_workspace, generate_timestamp_id, ensure_workspace_dir
from app.tool.backtest import BacktestTool
from app.logger import logger


class StrategyOptimizerTool(BaseTool):
    """Tool for optimizing trading strategy parameters."""

    name: str = "strategy_optimizer"
    description: str = (
        "优化交易策略参数以提高其性能。支持网格搜索、随机搜索和遗传算法等优化方法。"
        "可以针对年化收益率、最大回撤、夏普比率等指标进行优化。"
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "优化命令",
                "enum": [
                    "optimize_parameters",
                    "generate_optimization_report",
                    "get_optimization_status",
                ],
            },
            "strategy_file": {
                "type": "string",
                "description": "策略文件路径",
            },
            "data_file": {
                "type": "string",
                "description": "数据文件路径",
            },
            "start_date": {
                "type": "string",
                "description": "回测开始日期，格式为YYYY-MM-DD",
            },
            "end_date": {
                "type": "string",
                "description": "回测结束日期，格式为YYYY-MM-DD",
            },
            "initial_capital": {
                "type": "number",
                "description": "初始资金",
            },
            "commission": {
                "type": "number",
                "description": "交易佣金率（0.001表示0.1%）",
            },
            "optimization_method": {
                "type": "string",
                "description": "优化方法",
                "enum": ["grid_search", "random_search", "genetic_algorithm"],
            },
            "optimization_target": {
                "type": "string",
                "description": "优化目标指标",
                "enum": ["annual_return", "sharpe_ratio", "sortino_ratio", "max_drawdown", "profit_loss_ratio", "win_rate", "combined"],
            },
            "parameters_to_optimize": {
                "type": "object",
                "description": "需要优化的参数及其取值范围",
            },
            "max_iterations": {
                "type": "number",
                "description": "最大迭代次数（随机搜索和遗传算法）",
            },
            "optimization_id": {
                "type": "string",
                "description": "之前优化任务的ID",
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
        optimization_method: Optional[str] = "grid_search",
        optimization_target: Optional[str] = "sharpe_ratio",
        parameters_to_optimize: Optional[Dict] = None,
        max_iterations: Optional[int] = 100,
        optimization_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """执行策略优化命令。"""

        try:
            if command == "optimize_parameters":
                if not strategy_file or not data_file or not parameters_to_optimize:
                    return ToolResult(error="策略文件、数据文件和优化参数是必需的")

                # 转换日期格式如果提供
                start_date_obj = None
                end_date_obj = None

                if start_date:
                    try:
                        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
                    except ValueError:
                        return ToolResult(error=f"无效的开始日期格式: {start_date}，请使用YYYY-MM-DD格式")

                if end_date:
                    try:
                        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
                    except ValueError:
                        return ToolResult(error=f"无效的结束日期格式: {end_date}，请使用YYYY-MM-DD格式")

                # 运行参数优化
                result = self._optimize_parameters(
                    strategy_file=strategy_file,
                    data_file=data_file,
                    start_date=start_date_obj,
                    end_date=end_date_obj,
                    initial_capital=initial_capital,
                    commission=commission,
                    optimization_method=optimization_method,
                    optimization_target=optimization_target,
                    parameters_to_optimize=parameters_to_optimize,
                    max_iterations=max_iterations
                )

                return result

            elif command == "generate_optimization_report":
                if not optimization_id:
                    return ToolResult(error="生成优化报告需要优化任务ID")

                result = self._generate_optimization_report(optimization_id=optimization_id)
                return result

            elif command == "get_optimization_status":
                if not optimization_id:
                    return ToolResult(error="获取优化状态需要优化任务ID")

                result = self._get_optimization_status(optimization_id=optimization_id)
                return result

            else:
                return ToolResult(error=f"未知命令: {command}")

        except Exception as e:
            return ToolResult(error=f"执行命令 {command} 时出错: {str(e)}")

    def _optimize_parameters(
        self,
        strategy_file: str,
        data_file: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        optimization_method: str = "grid_search",
        optimization_target: str = "sharpe_ratio",
        parameters_to_optimize: Dict = {},
        max_iterations: int = 100
    ) -> ToolResult:
        """
        优化策略参数。
        优化结果将保存到工作空间。

        参数:
            strategy_file: 策略文件路径
            data_file: 数据文件路径
            start_date: 开始日期
            end_date: 结束日期
            initial_capital: 初始资金
            commission: 交易佣金率
            optimization_method: 优化方法
            optimization_target: 优化目标
            parameters_to_optimize: 需要优化的参数
            max_iterations: 最大迭代次数

        返回:
            包含优化结果信息的ToolResult对象
        """
        from app.tool.base import save_to_workspace, generate_timestamp_id, ensure_workspace_dir
        from app.logger import logger

        # 验证输入文件
        if not os.path.exists(strategy_file):
            return ToolResult(error=f"策略文件不存在: {strategy_file}")

        if not os.path.exists(data_file):
            return ToolResult(error=f"数据文件不存在: {data_file}")

        # 动态导入策略
        import importlib.util
        import sys

        try:
            # 创建唯一的模块名
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

        # 创建优化任务ID
        optimization_id = generate_timestamp_id("optimize")
        logger.info(f"开始参数优化任务 ID: {optimization_id}")

        # 确定评估指标函数
        def get_metric_value(results, metric):
            strat = results[0]

            if metric == "annual_return":
                returns = strat.analyzers.returns.get_analysis()
                return returns.get('ravg', 0.0)
            elif metric == "sharpe_ratio":
                sharpe = strat.analyzers.sharpe.get_analysis()
                return sharpe.get('sharperatio', 0.0)
            elif metric == "max_drawdown":
                drawdown = strat.analyzers.drawdown.get_analysis()
                # 对于最大回撤，我们返回负值，因为我们想最大化指标
                return -1.0 * drawdown.get('max', {}).get('drawdown', 0.0)
            elif metric == "profit_loss_ratio":
                custom = strat.analyzers.custom.get_analysis()
                return custom.get('profit_loss_ratio', 0.0)
            elif metric == "win_rate":
                custom = strat.analyzers.custom.get_analysis()
                return custom.get('win_rate', 0.0) / 100.0  # 转换为比例
            elif metric == "combined":
                # 组合多个指标
                returns = strat.analyzers.returns.get_analysis()
                sharpe = strat.analyzers.sharpe.get_analysis()
                drawdown = strat.analyzers.drawdown.get_analysis()
                custom = strat.analyzers.custom.get_analysis()

                annual_return = returns.get('ravg', 0.0)
                sharpe_ratio = sharpe.get('sharperatio', 0.0)
                max_dd = drawdown.get('max', {}).get('drawdown', 0.0)
                win_rate = custom.get('win_rate', 0.0) / 100.0
                pl_ratio = custom.get('profit_loss_ratio', 0.0)

                # 组合分数：年化收益 * 夏普比率 * 胜率 * 盈亏比 / 最大回撤
                # 为避免除以0，添加小值
                score = (annual_return * sharpe_ratio * win_rate * pl_ratio) / (max_dd + 0.01)
                return score
            else:
                return 0.0

        # 根据优化方法执行优化
        try:
            if optimization_method == "grid_search":
                # 生成参数网格
                param_grid = {}
                for param_name, param_range in parameters_to_optimize.items():
                    if isinstance(param_range, list):
                        param_grid[param_name] = param_range
                    elif isinstance(param_range, dict):
                        if "start" in param_range and "end" in param_range and "step" in param_range:
                            start = param_range["start"]
                            end = param_range["end"]
                            step = param_range["step"]
                            param_grid[param_name] = list(np.arange(start, end + step, step))
                        else:
                            return ToolResult(error=f"参数范围格式无效: {param_name}")
                    else:
                        return ToolResult(error=f"参数范围格式无效: {param_name}")

                # 生成所有参数组合
                param_names = list(param_grid.keys())
                param_values = list(itertools.product(*[param_grid[name] for name in param_names]))

                logger.info(f"参数优化: 共有 {len(param_values)} 种参数组合需要测试")

                # 存储优化结果
                optimization_results = []

                # 使用线程池并行运行回测
                def run_backtest_with_params(params_tuple):
                    params_dict = {name: value for name, value in zip(param_names, params_tuple)}

                    # 创建cerebro实例
                    cerebro = bt.Cerebro()

                    # 添加策略和参数
                    cerebro.addstrategy(strategy_class, **params_dict)

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
                    from app.tool.backtest import BacktestAnalyzer
                    cerebro.addanalyzer(BacktestAnalyzer, _name='custom')

                    # 运行回测
                    results = cerebro.run()

                    # 计算评估指标
                    metric_value = get_metric_value(results, optimization_target)

                    # 返回参数和指标值
                    return params_dict, metric_value, cerebro.broker.getvalue()

                # 使用线程池并行运行
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_results = {executor.submit(run_backtest_with_params, params): params for params in param_values}

                    # 进度跟踪
                    completed = 0
                    total = len(param_values)

                    for future in concurrent.futures.as_completed(future_results):
                        params_dict, metric_value, final_value = future.result()
                        optimization_results.append({
                            "parameters": params_dict,
                            "metric_value": metric_value,
                            "final_value": final_value
                        })

                        # 更新进度
                        completed += 1
                        if completed % max(1, total // 10) == 0:  # 每10%报告一次进度
                            logger.info(f"参数优化进度: {completed}/{total} ({completed/total*100:.1f}%)")

                # 按指标值排序结果
                optimization_results.sort(key=lambda x: x["metric_value"], reverse=True)

                logger.info(f"参数优化完成，找到 {len(optimization_results)} 个有效参数组合")

                # 准备结果对象
                output_results = {
                    "id": optimization_id,
                    "timestamp": datetime.now().isoformat(),
                    "strategy": os.path.basename(strategy_file),
                    "data": os.path.basename(data_file),
                    "optimization_method": optimization_method,
                    "optimization_target": optimization_target,
                    "parameters_to_optimize": parameters_to_optimize,
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                    "initial_capital": initial_capital,
                    "commission": commission,
                    "results": optimization_results,
                    "best_parameters": optimization_results[0]["parameters"] if optimization_results else None,
                    "best_metric_value": optimization_results[0]["metric_value"] if optimization_results else None,
                    "best_final_value": optimization_results[0]["final_value"] if optimization_results else None,
                    "status": "completed"
                }

                # 保存结果到工作空间
                results_file = f"{optimization_id}_results.json"
                results_file_path = save_to_workspace(output_results, results_file, "optimization_results", is_json=True)
                logger.info(f"优化结果已保存到: {results_file_path}")

                # 保存最佳参数的策略文件
                if optimization_results:
                    try:
                        # 读取原始策略文件
                        with open(strategy_file, 'r') as f:
                            strategy_code = f.read()

                        # 在策略文件顶部添加最佳参数作为注释
                        best_params = optimization_results[0]["parameters"]
                        best_params_str = json.dumps(best_params, ensure_ascii=False, indent=4)
                        optimized_code = f"""# 优化后的策略文件 - {optimization_id}
# 原始策略文件: {os.path.basename(strategy_file)}
# 优化方法: {optimization_method}
# 优化目标: {optimization_target}
# 最佳参数:
{best_params_str}

{strategy_code}
"""
                        # 保存带有最佳参数的优化策略
                        optimized_file = f"{optimization_id}_optimized_strategy.py"
                        optimized_path = save_to_workspace(optimized_code, optimized_file, "strategies")
                        logger.info(f"优化后的策略文件已保存到: {optimized_path}")

                        # 添加到输出结果
                        output_results["optimized_strategy_file"] = optimized_path

                        # 更新结果文件
                        save_to_workspace(output_results, results_file, "optimization_results", is_json=True)
                    except Exception as e:
                        logger.error(f"保存优化后的策略文件时出错: {str(e)}")

                # 返回摘要结果
                summary = {
                    "optimization_id": optimization_id,
                    "strategy": os.path.basename(strategy_file),
                    "optimization_method": optimization_method,
                    "optimization_target": optimization_target,
                    "total_combinations": len(param_values),
                    "best_parameters": optimization_results[0]["parameters"] if optimization_results else None,
                    "best_metric_value": optimization_results[0]["metric_value"] if optimization_results else None,
                    "best_final_value": optimization_results[0]["final_value"] if optimization_results else None,
                    "results_file": results_file_path,
                    "optimized_strategy_file": output_results.get("optimized_strategy_file"),
                    "message": f"参数优化完成。结果已保存到 {results_file_path}"
                }

                return ToolResult(output=json.dumps(summary, ensure_ascii=False, indent=2))

            elif optimization_method == "random_search":
                # 为简化起见，这里只实现网格搜索
                # 随机搜索和遗传算法在实际实现中会更复杂
                return ToolResult(error="随机搜索方法尚未实现，请使用网格搜索")

            elif optimization_method == "genetic_algorithm":
                return ToolResult(error="遗传算法方法尚未实现，请使用网格搜索")

            else:
                return ToolResult(error=f"未知的优化方法: {optimization_method}")

        except Exception as e:
            logger.exception(f"执行参数优化时出错: {str(e)}")
            return ToolResult(error=f"执行参数优化时出错: {str(e)}")

    def _generate_optimization_report(self, optimization_id: str) -> ToolResult:
        """生成优化报告。"""

        # 查找结果文件
        results_dir = os.path.join(os.getcwd(), "workspace", "optimization_results")
        results_file = os.path.join(results_dir, f"{optimization_id}.json")

        if not os.path.exists(results_file):
            return ToolResult(error=f"未找到优化结果文件: {results_file}")

        # 加载结果
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)

            # 提取最佳结果
            best_params = results.get("best_parameters", {})
            best_metric = results.get("best_metric_value", 0.0)
            best_final_value = results.get("best_final_value", 0.0)

            # 提取所有结果进行分析
            all_results = results.get("results", [])

            # 准备详细报告
            optimization_target = results.get("optimization_target", "")
            target_display_name = {
                "annual_return": "年化收益率",
                "sharpe_ratio": "夏普比率",
                "max_drawdown": "最大回撤",
                "profit_loss_ratio": "盈亏比",
                "win_rate": "胜率",
                "combined": "综合评分"
            }.get(optimization_target, optimization_target)

            # 参数敏感性分析
            parameter_sensitivity = {}

            if all_results:
                # 对每个参数进行分析
                for param_name in best_params.keys():
                    # 获取该参数的所有不同值
                    param_values = sorted(list(set([r["parameters"].get(param_name) for r in all_results])))

                    # 对于每个参数值，计算平均指标值
                    avg_metrics = []
                    for value in param_values:
                        results_with_value = [r for r in all_results if r["parameters"].get(param_name) == value]
                        avg_metric = sum([r["metric_value"] for r in results_with_value]) / len(results_with_value)
                        avg_metrics.append(avg_metric)

                    # 计算指标的变化范围来评估参数敏感性
                    sensitivity = max(avg_metrics) - min(avg_metrics) if avg_metrics else 0
                    parameter_sensitivity[param_name] = {
                        "values": param_values,
                        "avg_metrics": avg_metrics,
                        "sensitivity": sensitivity
                    }

            # 按敏感性排序参数
            sorted_params = sorted(parameter_sensitivity.items(), key=lambda x: x[1]["sensitivity"], reverse=True)

            # 生成参数推荐
            recommendations = []

            if sorted_params:
                most_sensitive_param = sorted_params[0][0]
                recommendations.append(f"参数 '{most_sensitive_param}' 对策略性能影响最大，应重点优化")

                # 为每个参数提供最佳值范围
                for param_name, sensitivity_data in sorted_params:
                    values = sensitivity_data["values"]
                    metrics = sensitivity_data["avg_metrics"]
                    best_value_index = metrics.index(max(metrics))
                    best_value = values[best_value_index]

                    recommendations.append(f"参数 '{param_name}' 的最佳值范围靠近 {best_value}")

            # 准备报告对象
            report = {
                "optimization_id": optimization_id,
                "strategy": results.get("strategy", ""),
                "optimization_method": results.get("optimization_method", ""),
                "optimization_target": {
                    "name": optimization_target,
                    "display_name": target_display_name
                },
                "date_range": {
                    "start_date": results.get("start_date", ""),
                    "end_date": results.get("end_date", "")
                },
                "best_result": {
                    "parameters": best_params,
                    "metric_value": best_metric,
                    "final_value": best_final_value,
                    "return_pct": (best_final_value / results.get("initial_capital", 100000.0) - 1.0) * 100.0
                },
                "parameter_sensitivity": {name: data for name, data in sorted_params},
                "recommendations": recommendations,
                "total_combinations_tested": len(all_results),
                "top_results": all_results[:10] if len(all_results) > 10 else all_results
            }

            return ToolResult(output=json.dumps(report, ensure_ascii=False, indent=2))

        except Exception as e:
            return ToolResult(error=f"生成优化报告时出错: {str(e)}")

    def _get_optimization_status(self, optimization_id: str) -> ToolResult:
        """获取优化任务的状态。"""

        # 查找结果文件
        results_dir = os.path.join(os.getcwd(), "workspace", "optimization_results")
        results_file = os.path.join(results_dir, f"{optimization_id}.json")

        if not os.path.exists(results_file):
            return ToolResult(error=f"未找到优化结果文件: {results_file}")

        # 加载结果
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)

            # 提取状态信息
            status = {
                "optimization_id": optimization_id,
                "strategy": results.get("strategy", ""),
                "status": results.get("status", "unknown"),
                "timestamp": results.get("timestamp", ""),
                "total_combinations": len(results.get("results", [])),
                "best_parameters": results.get("best_parameters", {}),
                "best_metric_value": results.get("best_metric_value", 0.0)
            }

            return ToolResult(output=json.dumps(status, ensure_ascii=False, indent=2))

        except Exception as e:
            return ToolResult(error=f"获取优化状态时出错: {str(e)}")


# 测试代码
if __name__ == "__main__":
    import asyncio

    async def test():
        tool = StrategyOptimizerTool()

        # 示例策略文件和数据文件
        strategy_file = "./momentum_strategy.py"
        data_file = "./data.csv"

        # 需要优化的参数
        parameters_to_optimize = {
            "rsi_period": {"start": 10, "end": 20, "step": 2},
            "rsi_overbought": {"start": 65, "end": 80, "step": 5},
            "rsi_oversold": {"start": 20, "end": 35, "step": 5}
        }

        # 运行参数优化
        result = await tool.execute(
            command="optimize_parameters",
            strategy_file=strategy_file,
            data_file=data_file,
            start_date="2020-01-01",
            end_date="2021-01-01",
            initial_capital=100000.0,
            commission=0.001,
            optimization_method="grid_search",
            optimization_target="sharpe_ratio",
            parameters_to_optimize=parameters_to_optimize
        )

        print(result.output or result.error)

        if result.output:
            # 提取优化ID
            result_data = json.loads(result.output)
            optimization_id = result_data.get("optimization_id")

            if optimization_id:
                # 生成优化报告
                report = await tool.execute(
                    command="generate_optimization_report",
                    optimization_id=optimization_id
                )

                print(report.output or report.error)

    asyncio.run(test())
