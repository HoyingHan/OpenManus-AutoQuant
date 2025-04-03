<p align="center">
  <img src="assets/logo.jpg" width="200"/>
</p>

English | [中文](README_zh.md) | [한국어](README_ko.md) | [日本語](README_ja.md)

[![GitHub stars](https://img.shields.io/github/stars/mannaandpoem/OpenManus?style=social)](https://github.com/mannaandpoem/OpenManus/stargazers)
&ensp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) &ensp;
[![Discord Follow](https://dcbadge.vercel.app/api/server/DYn29wFk9z?style=flat)](https://discord.gg/DYn29wFk9z)
[![Demo](https://img.shields.io/badge/Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/lyh-917/OpenManusDemo)

# 👋 OpenManus

Manus is incredible, but OpenManus can achieve any idea without an *Invite Code* 🛫!

Our team members [@Xinbin Liang](https://github.com/mannaandpoem) and [@Jinyu Xiang](https://github.com/XiangJinyu) (core authors), along with [@Zhaoyang Yu](https://github.com/MoshiQAQ), [@Jiayi Zhang](https://github.com/didiforgithub), and [@Sirui Hong](https://github.com/stellaHSR), we are from [@MetaGPT](https://github.com/geekan/MetaGPT). The prototype is launched within 3 hours and we are keeping building!

It's a simple implementation, so we welcome any suggestions, contributions, and feedback!

Enjoy your own agent with OpenManus!

We're also excited to introduce [OpenManus-RL](https://github.com/OpenManus/OpenManus-RL), an open-source project dedicated to reinforcement learning (RL)- based (such as GRPO) tuning methods for LLM agents, developed collaboratively by researchers from UIUC and OpenManus.

## Project Demo

<video src="https://private-user-images.githubusercontent.com/61239030/420168772-6dcfd0d2-9142-45d9-b74e-d10aa75073c6.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDEzMTgwNTksIm5iZiI6MTc0MTMxNzc1OSwicGF0aCI6Ii82MTIzOTAzMC80MjAxNjg3NzItNmRjZmQwZDItOTE0Mi00NWQ5LWI3NGUtZDEwYWE3NTA3M2M2Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTAzMDclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwMzA3VDAzMjIzOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTdiZjFkNjlmYWNjMmEzOTliM2Y3M2VlYjgyNDRlZDJmOWE3NWZhZjE1MzhiZWY4YmQ3NjdkNTYwYTU5ZDA2MzYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.UuHQCgWYkh0OQq9qsUWqGsUbhG3i9jcZDAMeHjLt5T4" data-canonical-src="https://private-user-images.githubusercontent.com/61239030/420168772-6dcfd0d2-9142-45d9-b74e-d10aa75073c6.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDEzMTgwNTksIm5iZiI6MTc0MTMxNzc1OSwicGF0aCI6Ii82MTIzOTAzMC80MjAxNjg3NzItNmRjZmQwZDItOTE0Mi00NWQ5LWI3NGUtZDEwYWE3NTA3M2M2Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTAzMDclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwMzA3VDAzMjIzOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTdiZjFkNjlmYWNjMmEzOTliM2Y3M2VlYjgyNDRlZDJmOWE3NWZhZjE1MzhiZWY4YmQ3NjdkNTYwYTU5ZDA2MzYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.UuHQCgWYkh0OQq9qsUWqGsUbhG3i9jcZDAMeHjLt5T4" controls="controls" muted="muted" class="d-block rounded-bottom-2 border-top width-fit" style="max-height:640px; min-height: 200px"></video>

## Installation

We provide two installation methods. Method 2 (using uv) is recommended for faster installation and better dependency management.

### Method 1: Using conda

1. Create a new conda environment:

```bash
conda create -n open_manus python=3.12
conda activate open_manus
```

2. Clone the repository:

```bash
git clone https://github.com/mannaandpoem/OpenManus.git
cd OpenManus
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Method 2: Using uv (Recommended)

1. Install uv (A fast Python package installer and resolver):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:

```bash
git clone https://github.com/mannaandpoem/OpenManus.git
cd OpenManus
```

3. Create a new virtual environment and activate it:

```bash
uv venv --python 3.12
source .venv/bin/activate  # On Unix/macOS
# Or on Windows:
# .venv\Scripts\activate
```

4. Install dependencies:

```bash
uv pip install -r requirements.txt
```

### Browser Automation Tool (Optional)
```bash
playwright install
```

## Configuration

OpenManus requires configuration for the LLM APIs it uses. Follow these steps to set up your configuration:

1. Create a `config.toml` file in the `config` directory (you can copy from the example):

```bash
cp config/config.example.toml config/config.toml
```

2. Edit `config/config.toml` to add your API keys and customize settings:

```toml
# Global LLM configuration
[llm]
model = "gpt-4o"
base_url = "https://api.openai.com/v1"
api_key = "sk-..."  # Replace with your actual API key
max_tokens = 4096
temperature = 0.0

# Optional configuration for specific LLM models
[llm.vision]
model = "gpt-4o"
base_url = "https://api.openai.com/v1"
api_key = "sk-..."  # Replace with your actual API key
```

## Quick Start

One line for run OpenManus:

```bash
python main.py
```

Then input your idea via terminal!

For MCP tool version, you can run:
```bash
python run_mcp.py
```

For unstable multi-agent version, you also can run:

```bash
python run_flow.py
```

## How to contribute

We welcome any friendly suggestions and helpful contributions! Just create issues or submit pull requests.

Or contact @mannaandpoem via 📧email: mannaandpoem@gmail.com

**Note**: Before submitting a pull request, please use the pre-commit tool to check your changes. Run `pre-commit run --all-files` to execute the checks.

## Community Group
Join our networking group on Feishu and share your experience with other developers!

<div align="center" style="display: flex; gap: 20px;">
    <img src="assets/community_group.jpg" alt="OpenManus 交流群" width="300" />
</div>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mannaandpoem/OpenManus&type=Date)](https://star-history.com/#mannaandpoem/OpenManus&Date)

## Sponsors
Thanks to [PPIO](https://ppinfra.com/user/register?invited_by=OCPKCN&utm_source=github_openmanus&utm_medium=github_readme&utm_campaign=link) for computing source support.
> PPIO: The most affordable and easily-integrated MaaS and GPU cloud solution.


## Acknowledgement

Thanks to [anthropic-computer-use](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo)
and [browser-use](https://github.com/browser-use/browser-use) for providing basic support for this project!

Additionally, we are grateful to [AAAJ](https://github.com/metauto-ai/agent-as-a-judge), [MetaGPT](https://github.com/geekan/MetaGPT), [OpenHands](https://github.com/All-Hands-AI/OpenHands) and [SWE-agent](https://github.com/SWE-agent/SWE-agent).

We also thank stepfun(阶跃星辰) for supporting our Hugging Face demo space.

OpenManus is built by contributors from MetaGPT. Huge thanks to this agent community!

## Cite
```bibtex
@misc{openmanus2025,
  author = {Xinbin Liang and Jinyu Xiang and Zhaoyang Yu and Jiayi Zhang and Sirui Hong},
  title = {OpenManus: An open-source framework for building general AI agents},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/mannaandpoem/OpenManus}},
}
```

# OpenManus-AutoQuant (自动量化交易系统)

OpenManus-AutoQuant 是一个自动化的量化交易分析系统，能够自动获取股票数据、分析市场状况、生成交易策略、执行回测并优化策略参数。

## 主要功能

1. **股票数据获取**：自动从AKShare获取股票的历史价格数据和基本面数据
2. **技术指标分析**：计算并分析多种技术指标（RSI、MACD、布林带等）
3. **市场状况评估**：根据技术指标和基本面数据评估市场环境
4. **策略生成**：根据分析结果自动生成适合当前市场状况的交易策略代码
5. **回测**：使用Backtrader框架对生成的策略进行回测
6. **策略优化**：自动优化策略参数以提高性能

## 支持的策略类型

- 动量策略（Momentum）
- 趋势跟踪（Trend Following）
- 均值回归（Mean Reversion）
- 双均线交叉（Dual MA Crossover）
- MACD策略
- RSI策略
- 布林带策略
- 自定义组合策略

## 系统架构

```
app/
├── main.py              # 主程序
├── tool/
│   ├── base.py          # 基础工具类
│   ├── stock_data_fetch.py    # 股票数据获取工具
│   ├── stock_analysis.py      # 股票分析工具
│   ├── strategy_generator.py  # 策略生成工具
│   ├── backtest.py      # 回测工具
│   └── strategy_optimizer.py  # 策略优化工具
└── utils/
    └── data_utils.py    # 数据处理工具函数
```

## 安装指南

### 前置条件

- Python 3.8+
- pip（Python包管理器）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 依赖项

- akshare：获取股票数据
- pandas：数据处理
- numpy：数值计算
- matplotlib：可视化图表
- backtrader：回测框架
- talib：技术指标计算
- scikit-learn：机器学习算法（用于高级分析）

## 使用方法

### 基本用法

```python
import asyncio
from app.main import AutoQuantSystem

async def example():
    system = AutoQuantSystem()

    # 运行完整流程
    results = await system.run_full_pipeline(
        stock_code="000001",    # 股票代码
        start_date="2020-01-01", # 开始日期
        end_date="2022-12-31",   # 结束日期
        strategy_type="momentum",  # 策略类型
        optimize_strategy=True     # 是否优化策略
    )

    print(results)

# 运行示例
asyncio.run(example())
```

### 分步骤使用

```python
import asyncio
import json
from app.main import AutoQuantSystem

async def step_by_step_example():
    system = AutoQuantSystem()

    # 1. 获取股票数据
    data = await system.fetch_stock_data(
        stock_code="000001",
        start_date="2020-01-01",
        end_date="2022-12-31"
    )
    data_file = data["history"]["csv_file"]

    # 2. 分析股票
    analysis = await system.analyze_stock(
        stock_code="000001",
        data_file=data_file
    )

    # 3. 生成策略
    strategy = await system.generate_strategy(
        strategy_type="trend_following",
        analysis_results=analysis
    )
    strategy_file = strategy["strategy_file"]

    # 4. 运行回测
    backtest = await system.run_backtest(
        strategy_file=strategy_file,
        data_file=data_file,
        start_date="2020-01-01",
        end_date="2022-12-31"
    )

    # 5. 优化策略
    parameters_to_optimize = {
        "fast_period": {"start": 10, "end": 30, "step": 5},
        "slow_period": {"start": 30, "end": 60, "step": 10}
    }

    optimization = await system.optimize_strategy(
        strategy_file=strategy_file,
        data_file=data_file,
        start_date="2020-01-01",
        end_date="2022-12-31",
        parameters_to_optimize=parameters_to_optimize
    )

    print(json.dumps({
        "data": data,
        "analysis": analysis,
        "strategy": strategy,
        "backtest": backtest,
        "optimization": optimization
    }, indent=2))

# 运行示例
asyncio.run(step_by_step_example())
```

## 工具使用说明

### 股票数据获取工具 (StockDataFetchTool)

提供以下功能：

- `get_stock_history`: 获取股票历史价格数据
- `get_stock_fundamental`: 获取股票基本面数据
- `get_index_data`: 获取指数数据
- `search_stock`: 搜索股票信息

### 股票分析工具 (StockAnalysisTool)

提供以下功能：

- `analyze_stock`: 全面分析股票
- `calculate_indicators`: 计算技术指标
- `analyze_market_condition`: 分析市场环境
- `generate_trading_signals`: 生成交易信号

### 策略生成工具 (StrategyGeneratorTool)

提供以下功能：

- `generate_strategy`: 生成交易策略
- `get_strategy_template`: 获取策略模板
- `list_available_strategies`: 列出可用策略类型

### 回测工具 (BacktestTool)

提供以下功能：

- `run_backtest`: 运行回测
- `analyze_results`: 分析回测结果
- `plot_performance`: 绘制性能图表

### 策略优化工具 (StrategyOptimizerTool)

提供以下功能：

- `optimize_parameters`: 优化策略参数
- `generate_optimization_report`: 生成优化报告
- `get_optimization_status`: 获取优化状态

## 示例输出

策略回测报告示例：

```json
{
  "result_id": "backtest_20230501123456",
  "strategy": "momentum_strategy.py",
  "data": "000001.csv",
  "summary": {
    "final_value": 125463.25,
    "total_return_pct": 25.46,
    "annual_return": 12.15,
    "sharpe_ratio": 1.32,
    "max_drawdown": 8.75,
    "total_trades": 42,
    "win_rate": 68.5,
    "profit_loss_ratio": 2.1
  }
}
```

## 注意事项

1. 数据获取需要有稳定的网络连接
2. 回测结果不代表未来表现
3. 使用实盘交易前请充分测试策略

## 后续开发计划

1. 添加机器学习模型预测股价走势
2. 支持更多数据源和国际市场
3. 开发图形用户界面
4. 增加实盘交易接口
5. 添加风险管理模块

## 贡献指南

欢迎提交Pull Request或Issue来帮助改进项目。

## 许可证

MIT License
