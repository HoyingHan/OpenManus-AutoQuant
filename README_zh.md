
[English](README.md) | 中文


# 👋 OpenManus-AutoQuant（自迭代金融量化Agent）

## 项目描述

传统的金融量化的工作流程是：分析数据 → 构造因子 → 编写策略代码 → 回测代码 → 优化策略直到达到回测标准，开发周期长且依赖人工经验，难以应对高频市场变化。

OpenManus-AutoQuant 的目标是：将上述过程自动化，由AI构思策略，写代码，回测，优化再回测，直到满足最终目标。核心在于自生成、自迭代、自优化。

## 项目技术亮点

1. **精准数据管道**：集成 akshare 股票数据源，通过结构化接口文档与数据校验模块，确保数据质量与准确。
2. **数据分析**：针对股票数据的特性做了业务性增强。
3. **策略生成**：基于分析结果动态生成策略。
4. **智能回测**：整合 backtrader 框架，支持多维度评估（Sharpe比率、最大回撤、胜率）。
5. **策略优化**：基于回测数据对策略进行优化。

## 应用场景与价值

- **私募基金**：快速生成备选策略池，降低研究员人力成本；
- **个人投资者**：通过自然语言输入投资偏好，自动生成定制策略；
- **学术研究**：提供策略生成过程的思维链日志，辅助市场行为学研究。

## 安装指南

我们提供两种安装方式。推荐使用方式二（uv），因为它能提供更快的安装速度和更好的依赖管理。

### 方式一：使用 conda

1. 创建新的 conda 环境：

```bash
conda create -n open_manus python=3.12
conda activate open_manus
```

2. 克隆仓库：

```bash
git clone https://github.com/mannaandpoem/OpenManus.git
cd OpenManus
```

3. 安装依赖：

```bash
pip install -r requirements.txt
```

### 方式二：使用 uv（推荐）

1. 安装 uv（一个快速的 Python 包管理器）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. 克隆仓库：

```bash
git clone https://github.com/lumina-lumi/OpenManus-AutoQuant.git
cd OpenManus-AutoQuant
```

3. 创建并激活虚拟环境：

```bash
uv venv --python 3.12
source .venv/bin/activate  # Unix/macOS 系统
# Windows 系统使用：
# .venv\Scripts\activate
```

4. 安装依赖：

```bash
uv pip install -r requirements.txt

uv pip install -r requirements.txt --index-url https://mirrors.aliyun.com/pypi/simple/
```

### 浏览器自动化工具（可选）
```bash
playwright install
```

