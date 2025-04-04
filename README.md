

English | [中文](README_zh.md)



# 👋 OpenManus-AutoQuant (Self-iterative Financial Quantitative Agent)

## Project Description

Traditional financial quantitative workflows involve: data analysis → factor construction → strategy code writing → backtesting → strategy optimization until meeting backtesting standards. This process is time-consuming, relies heavily on human expertise, and struggles to adapt to high-frequency market changes.

OpenManus-AutoQuant aims to automate this entire process, with AI conceptualizing strategies, writing code, backtesting, optimizing, and re-testing until the final goal is met. The core value lies in self-generation, self-iteration, and self-optimization.

## Technical Highlights

1. **Precise Data Pipeline**: Integrates akshare stock data sources with structured API documentation and data validation modules to ensure data quality and accuracy.
2. **Data Analysis**: Enhanced business-specific analysis for stock data characteristics.
3. **Strategy Generation**: Dynamically generates strategies based on analysis results.
4. **Intelligent Backtesting**: Integrates the backtrader framework, supporting multi-dimensional evaluation (Sharpe ratio, maximum drawdown, win rate).
5. **Strategy Optimization**: Optimizes strategies based on backtesting data.

## Application Scenarios & Value

- **Private Equity Funds**: Quickly generate candidate strategy pools, reducing analyst labor costs.
- **Individual Investors**: Automatically generate customized strategies through natural language input of investment preferences.
- **Academic Research**: Provide chain-of-thought logs of the strategy generation process to assist market behavioral research.

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
git clone https://github.com/lumina-lumi/OpenManus-AutoQuant.git
cd OpenManus-AutoQuant
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


