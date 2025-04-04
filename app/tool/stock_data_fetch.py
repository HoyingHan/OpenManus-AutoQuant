"""
股票数据获取工具。

包含两个工具：
1. StockAPIAnalyzerTool: 分析用户需求，推荐合适的akshare API
2. StockCodeGeneratorTool: 根据选定的API生成并执行代码获取数据
"""

import inspect
import json
import os
import tempfile
import re
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging
import time
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.tool.base import BaseTool, ToolResult, save_to_workspace, generate_timestamp_id, ensure_workspace_dir
from app.logger import logger


class StockAPIAnalyzerTool(BaseTool):
    """分析用户需求并推荐适合的akshare API的工具。"""

    name: str = "stock_api_analyzer"
    description: str = (
        "分析用户需求并推荐适合的akshare API。"
        "帮助识别合适的数据源用于股票分析任务。"
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "用户的数据需求描述（自然语言）",
            },
            "data_type": {
                "type": "string",
                "description": "需要的数据类型（如'stock_price', 'financial_report', 'market_overview'）",
                "enum": [
                    "stock_price",
                    "financial_report",
                    "market_overview",
                    "industry_data",
                    "concept_board",
                    "fund_flow",
                    "stock_ranking",
                    "research_report",
                    "other"
                ],
            },
            "market": {
                "type": "string",
                "description": "感兴趣的市场（如'A股', '港股', '美股'）",
                "enum": ["A股", "港股", "美股", "全球"],
            },
            "time_period": {
                "type": "string",
                "description": "感兴趣的时间段（如'historical', 'real_time'）",
                "enum": ["historical", "real_time", "both"],
            },
        },
        "required": ["query"],
    }

    async def execute(
        self,
        query: str,
        data_type: Optional[str] = None,
        market: Optional[str] = None,
        time_period: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """
        执行API分析器来推荐合适的akshare API。

        参数:
            query: 用户查询（自然语言描述需求）
            data_type: 数据类型
            market: 市场类型
            time_period: 时间周期

        返回:
            包含推荐API的ToolResult对象
        """
        try:
            # 加载API概览
            api_overview_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resource", "api_overview.md")
            if not os.path.exists(api_overview_path):
                return ToolResult(error=f"API概览文件未找到: {api_overview_path}")

            with open(api_overview_path, "r", encoding="utf-8") as f:
                api_overview_content = f.read()

            # 解析API概览为结构化数据
            api_data = self._parse_api_overview(api_overview_content)

            # 根据用户查询和参数过滤API
            matched_apis = self._match_apis(api_data, query, data_type, market, time_period)

            # 格式化结果
            if not matched_apis:
                return ToolResult(output="未找到匹配您需求的API。请提供更具体的细节。")

            result = {
                "query": query,
                "data_type": data_type,
                "market": market,
                "time_period": time_period,
                "recommended_apis": matched_apis,
                "total_apis_found": len(matched_apis)
            }

            return ToolResult(output=json.dumps(result, ensure_ascii=False, indent=2))

        except Exception as e:
            logging.exception("API分析过程中出错")
            return ToolResult(error=f"分析API需求时出错: {str(e)}")

    def _parse_api_overview(self, content: str) -> List[Dict]:
        """
        将API概览markdown解析为结构化数据。

        参数:
            content: API概览markdown内容

        返回:
            API信息列表
        """
        apis = []
        current_category = None

        # 按部分拆分内容
        lines = content.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # 类别标题
            if line.startswith('## '):
                current_category = line[3:].strip()
                i += 1
                continue

            # 表格标题
            if '| 分类 | 接口名 | 描述 |' in line or '|------|--------|------|' in line:
                # 跳过标题行和分隔行
                i += 2
                continue

            # 处理表格行
            if line.startswith('|') and '|' in line[1:]:
                row = line.strip('|').split('|')
                if len(row) >= 3:
                    category = row[0].strip()
                    api_name = row[1].strip()
                    description = row[2].strip()

                    apis.append({
                        "main_category": current_category,
                        "sub_category": category,
                        "api_name": api_name,
                        "description": description
                    })
            i += 1

        return apis

    def _match_apis(
        self,
        api_data: List[Dict],
        query: str,
        data_type: Optional[str] = None,
        market: Optional[str] = None,
        time_period: Optional[str] = None
    ) -> List[Dict]:
        """
        根据用户查询和参数匹配API。

        参数:
            api_data: 可用API列表
            query: 用户查询
            data_type: 数据类型
            market: 市场类型
            time_period: 时间周期

        返回:
            匹配的API列表，按相关性评分排序
        """
        # 从查询中提取关键信息
        if not data_type and "股价" in query:
            data_type = "stock_price"
        if not market and ("A股" in query or "a股" in query):
            market = "A股"
        if not time_period and "历史" in query:
            time_period = "historical"

        # 提取股票名称
        stock_name = None
        stock_patterns = [r'(\w+)的(历史)?(股价|行情)', r'(历史)?(股价|行情)数据.*?(\w+)']
        for pattern in stock_patterns:
            match = re.search(pattern, query)
            if match:
                groups = [g for g in match.groups() if g and g not in ["历史", "股价", "行情"]]
                if groups:
                    stock_name = groups[0]
                    break

        # 市场关键词映射
        market_keywords = {
            "A股": ["a股", "a", "沪深", "沪", "深", "上证", "深证", "上海", "深圳", "主板"],
            "港股": ["港股", "港", "香港", "hk"],
            "美股": ["美股", "美", "us", "纳斯达克", "纽约"],
            "全球": ["全球", "global", "世界", "国际"]
        }

        # 数据类型关键词映射
        data_type_keywords = {
            "stock_price": ["价格", "股价", "行情", "k线", "分时", "走势", "涨跌", "历史股价", "历史行情"],
            "financial_report": ["财报", "报表", "资产负债", "利润", "现金流", "财务"],
            "market_overview": ["大盘", "指数", "市场", "板块", "概况", "总览"],
            "industry_data": ["行业", "板块", "产业", "sector"],
            "concept_board": ["概念", "题材", "热点"],
            "fund_flow": ["资金", "流向", "资金流", "主力", "净流入", "净流出"],
            "stock_ranking": ["排名", "榜单", "热门", "排行"],
            "research_report": ["研报", "分析", "研究", "评级", "目标价"]
        }

        # 时间周期关键词
        time_keywords = {
            "historical": ["历史", "过去", "前", "日线", "周线", "月线", "季度", "年度", "日k", "周k", "月k"],
            "real_time": ["实时", "盘中", "当前", "最新", "分时", "分钟", "即时", "盘口"]
        }

        matched_apis = []

        # 根据股票名称和市场，增加特定的API匹配
        if stock_name == "贵州茅台" or "茅台" in query:
            special_apis = ["stock_zh_a_hist", "stock_zh_a_daily"]
            for api_name in special_apis:
                for api in api_data:
                    if api["api_name"] == api_name:
                        api_copy = api.copy()
                        api_copy["relevance_score"] = 5  # 高优先级匹配
                        matched_apis.append(api_copy)

        # 常规匹配逻辑
        for api in api_data:
            score = 0
            combined_text = f"{api['main_category']} {api['sub_category']} {api['description']}".lower()

            # 增加对A股历史数据的匹配权重
            if "历史" in query and "A股" in api["main_category"] and "历史" in combined_text:
                score += 2

            # 增加对特定API的匹配权重
            if "hist" in api["api_name"] and "历史" in query:
                score += 2

            # 增加对常用API的匹配权重
            common_apis = ["stock_zh_a_hist", "stock_zh_a_daily", "stock_zh_index_daily"]
            if api["api_name"] in common_apis:
                score += 1

            # 根据查询关键词匹配
            query_keywords = query.lower().split()
            for keyword in query_keywords:
                if keyword in combined_text or keyword in api['api_name'].lower():
                    score += 1
                # 增加部分匹配
                elif len(keyword) > 2:
                    for word in combined_text.split():
                        if keyword in word:
                            score += 0.5
                            break

            # 根据市场匹配
            if market:
                market_match = False
                for kw in market_keywords.get(market, []):
                    if kw in combined_text:
                        market_match = True
                        score += 2
                        break

                # 如果指定了市场但不匹配，则跳过
                if not market_match and market != "全球" and "A股" not in api["main_category"]:
                    continue

            # 根据数据类型匹配
            if data_type:
                data_type_match = False
                for kw in data_type_keywords.get(data_type, []):
                    if kw in combined_text or kw in query.lower():
                        data_type_match = True
                        score += 2
                        break

                # 降低评分但不完全跳过
                if not data_type_match:
                    score -= 1

            # 根据时间周期匹配
            if time_period and time_period != "both":
                time_match = False
                for kw in time_keywords.get(time_period, []):
                    if kw in combined_text or kw in query.lower():
                        time_match = True
                        score += 1
                        break

                # 降低评分但不完全跳过
                if not time_match:
                    score -= 1

            # 如果评分为正，添加到结果中
            if score > 0:
                api_copy = api.copy()
                api_copy["relevance_score"] = score
                matched_apis.append(api_copy)

        # 确保返回结果不为空
        if not matched_apis and "A股" in query and "历史" in query:
            # 当没有匹配但查询明确要求A股历史数据时，返回最常用的A股历史数据API
            default_apis = ["stock_zh_a_hist", "stock_zh_a_daily"]
            for api in api_data:
                if api["api_name"] in default_apis:
                    api_copy = api.copy()
                    api_copy["relevance_score"] = 3
                    matched_apis.append(api_copy)

        # 按相关性评分排序
        matched_apis.sort(key=lambda x: x["relevance_score"], reverse=True)

        # 去除重复的API
        unique_apis = []
        api_names = set()
        for api in matched_apis:
            if api["api_name"] not in api_names:
                unique_apis.append(api)
                api_names.add(api["api_name"])

        # 限制为前10个结果以保持清晰
        return unique_apis[:10]


class StockCodeGeneratorTool(BaseTool):
    """根据akshare API生成并执行代码的工具。"""

    name: str = "stock_code_generator"
    description: str = (
        "根据API文档为选定的akshare API生成和执行Python代码。"
        "基于生成的代码检索数据。"
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "api_name": {
                "type": "string",
                "description": "要使用的akshare API名称（例如，'stock_zh_a_hist'）",
            },
            "params": {
                "type": "object",
                "description": "传递给API函数的参数",
            },
            "plot": {
                "type": "boolean",
                "description": "是否生成数据可视化",
            },
            "execute": {
                "type": "boolean",
                "description": "如果为true，执行生成的代码；如果为false，只生成不执行",
            },
        },
        "required": ["api_name"],
    }

    async def execute(
        self,
        api_name: str = None,
        params: dict = None,
        execute: bool = True,
        plot: bool = False,
        input_data: str = None,
        **kwargs
    ) -> ToolResult:
        """
        根据指定的API生成并可选执行代码。

        参数可以通过两种方式传入：
        1. 直接传递 api_name, params 等参数
        2. 通过 input_data 字符串(JSON格式)传递

        返回:
            生成的代码及其执行结果（如果选择执行）
        """
        try:
            # 方式1：直接使用传入的参数
            if api_name:
                execute_code = execute  # 变量重命名，避免与参数名冲突
                params = params or {}
            # 方式2：从input_data解析参数
            elif input_data:
                try:
                    data = json.loads(input_data)
                    api_name = data.get("api_name")
                    params = data.get("params", {})
                    execute_code = data.get("execute", True)
                    plot = data.get("plot", False)
                except json.JSONDecodeError:
                    # 如果input_data不是JSON，则直接将其作为API名称
                    api_name = input_data
                    params = {}
                    execute_code = True
                    plot = False
            else:
                return ToolResult(error="未提供API名称")

            # 获取API文档
            api_doc = self._find_api_documentation(api_name)
            if not api_doc:
                # 尝试在输入是API名称之前包含其他数据的情况
                if isinstance(api_name, str) and "stock_" in api_name:
                    api_parts = re.findall(r'(stock_[a-zA-Z0-9_]+)', api_name)
                    if api_parts:
                        for part in api_parts:
                            api_doc = self._find_api_documentation(part)
                            if api_doc:
                                api_name = part
                                break

            # 如果仍然找不到API文档
            if not api_doc:
                return ToolResult(error=f"错误: 在stockapi目录中未找到API '{api_name}'的文档")

            # 检查是否存在日期参数，并记录提示信息
            date_param_warning = ""
            if params:
                # 识别可能的日期参数并记录
                potential_date_params = []
                for key, value in params.items():
                    if isinstance(value, str) and any(date_keyword in key.lower()
                                                     for date_keyword in ["date", "time", "start", "end", "日期", "时间", "开始", "结束"]):
                        potential_date_params.append(key)

                        # 检查日期格式，如果不是YYYYMMDD格式，则记录警告
                        if not re.match(r'^\d{8}$', value) and re.match(r'^\d{4}-\d{2}-\d{2}$', value):
                            logger.warning(f"日期参数 '{key}' 使用了YYYY-MM-DD格式 ('{value}')，但akshare需要YYYYMMDD格式")
                            date_param_warning += f"注意: 参数 '{key}' 的日期格式应为'YYYYMMDD'，当前值为'{value}'。生成的代码将自动处理此问题。\n"

                if potential_date_params:
                    logger.info(f"发现以下可能的日期参数: {', '.join(potential_date_params)}")
                    if not date_param_warning:
                        logger.info("所有日期参数格式已正确 (YYYYMMDD)")

            # 判断是否为历史数据API并需要循环处理
            is_historical_api = (
                "hist" in api_name.lower() or
                "daily" in api_name.lower() or
                "历史" in api_doc.get("content", "").lower()
            )

            start_date_param = None
            end_date_param = None

            if is_historical_api and params:
                for key in params.keys():
                    if any(keyword in key.lower() for keyword in ["start", "begin", "from", "开始"]):
                        start_date_param = key
                    elif any(keyword in key.lower() for keyword in ["end", "to", "结束"]):
                        end_date_param = key

            historical_notice = ""
            if is_historical_api:
                historical_notice = (
                    "注意: 检测到您正在获取历史数据，系统将自动获取当前日期往前约10个交易日的数据。"
                    "如果您提供了起始日期和结束日期参数，它们将被自动替换为当前日期区间。"
                )
                logger.info("历史数据获取已配置为获取最近10个交易日的数据")

            # 生成代码
            generated_code = self._generate_code(api_name, api_doc, params, plot)

            result = {
                "api_name": api_name,
                "generated_code": generated_code,
                "message": "代码生成成功。未执行。"
            }

            # 如果有日期参数警告，添加到结果中
            if date_param_warning:
                result["warning"] = date_param_warning.strip()

            # 添加历史数据获取说明
            if historical_notice:
                result["historical_notice"] = historical_notice

            # 如果需要执行代码
            if execute_code:
                try:
                    execution_result = self._execute_generated_code(generated_code)
                    result["execution_result"] = execution_result
                    result["has_plot"] = "plot_path" in execution_result if isinstance(execution_result, dict) else False
                except Exception as e:
                    result["execution_result"] = {"status": "error", "error": str(e), "message": "代码执行失败"}
                    result["has_plot"] = False

            return ToolResult(output=json.dumps(result, ensure_ascii=False))

        except Exception as e:
            logging.exception("生成或执行代码时出错")
            return ToolResult(error=f"生成或执行代码时出错: {str(e)}")

    def _find_api_documentation(self, api_name: str) -> Optional[Dict]:
        """
        查找指定API的详细文档

        参数:
            api_name: API名称

        返回:
            包含API文档的字典，如果未找到则返回None
        """
        stockapi_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resource", "stockapi")

        # 尝试查找所有可能的文档位置
        potential_files = []

        # 直接搜索文件
        for file in os.listdir(stockapi_dir):
            file_path = os.path.join(stockapi_dir, file)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if f"接口: {api_name}" in content:
                        potential_files.append(file_path)

        if not potential_files:
            logging.warning(f"API '{api_name}' 文档未找到")
            return None

        # 使用第一个匹配的文件
        file_path = potential_files[0]

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

            # 查找文档内容
            params_section = ""
            lines = content.split('\n')
            header_found = False
            for i, line in enumerate(lines):
                if f"接口: {api_name}" in line:
                    header_found = True
                    # 向后查找参数部分
                    for j in range(i, len(lines)):
                        if "输入参数" in lines[j]:
                            # 继续向后查找，直到下一个接口或文档结束
                            params_start = j
                            params_end = len(lines)
                            for k in range(j + 1, len(lines)):
                                if "接口: " in lines[k]:
                                    params_end = k
                                    break
                            params_section = '\n'.join(lines[params_start:params_end])
                            break
                    break

            if header_found:
                return {
                    "name": api_name,
                    "file_path": file_path,
                    "content": content,
                    "params": params_section
                }

            return None

    def _generate_code(
        self,
        api_name: str,
        api_doc: Dict,
        params: Optional[Dict] = None,
        plot: bool = False
    ) -> str:
        """
        根据文档为指定的API生成Python代码。

        参数:
            api_name: API名称
            api_doc: API文档
            params: API参数
            plot: 是否生成可视化

        返回:
            生成的Python代码
        """
        code_lines = [
            "import akshare as ak",
            "import pandas as pd",
            "import json",
            "import tempfile",
            "import re",
            "from datetime import date, datetime, timedelta",
            "",
            "# Define JSON encoder for datetime/date objects",
            "class DateTimeEncoder(json.JSONEncoder):",
            "    def default(self, obj):",
            "        if isinstance(obj, (datetime, date)):",
            "            return obj.isoformat()",
            "        return super().default(obj)",
            "",
            "# Format date parameters for akshare",
            "def format_date_param(date_str):",
            "    \"\"\"将日期字符串转换为akshare所需的'YYYYMMDD'格式\"\"\"",
            "    # 检查是否已经是YYYYMMDD格式",
            "    if re.match(r'^\\d{8}$', str(date_str)):",
            "        return date_str",
            "    # 尝试转换YYYY-MM-DD格式",
            "    elif re.match(r'^\\d{4}-\\d{2}-\\d{2}$', str(date_str)):",
            "        return str(date_str).replace('-', '')",
            "    # 尝试解析任何日期格式并转换",
            "    try:",
            "        dt = pd.to_datetime(date_str)",
            "        return dt.strftime('%Y%m%d')",
            "    except:",
            "        # 无法处理则原样返回",
            "        return date_str",
            "",
        ]

        if plot:
            code_lines.extend([
                "import plotly.graph_objects as go",
                "from plotly.subplots import make_subplots",
                ""
            ])

        # 从文档中提取参数（如果可用）
        param_info = {}
        if api_doc.get("content"):
            # 尝试从文档中解析参数
            # 这是一个简单的方法 - 在实际实现中，你会希望更复杂的解析
            param_lines = []
            in_param_section = False

            for line in api_doc["content"].split("\n"):
                if "参数" in line and ("名称" in line or "说明" in line):
                    in_param_section = True
                    continue

                if in_param_section:
                    if line.strip() == "" or "---" in line:
                        in_param_section = False
                        continue

                    param_lines.append(line)

            # 解析参数行并确定日期参数
            date_params = []
            for line in param_lines:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 2:
                    param_name = parts[0]
                    param_desc = parts[1] if len(parts) > 1 else ""
                    param_info[param_name] = {"description": param_desc}

                    # 识别可能的日期参数
                    if any(keyword in param_name.lower() or keyword in param_desc.lower()
                           for keyword in ["date", "日期", "time", "时间", "start", "end", "开始", "结束"]):
                        date_params.append(param_name)


        # 判断是否为历史数据API
        is_historical_api = (
            "hist" in api_name.lower() or
            "daily" in api_name.lower() or
            "历史" in api_doc.get("content", "").lower() or
            bool(date_params)
        )

        # 确定日期参数名称
        start_date_param = None
        end_date_param = None

        if is_historical_api and params:
            for key in params.keys():
                if any(keyword in key.lower() for keyword in ["start", "begin", "from", "开始"]):
                    start_date_param = key
                elif any(keyword in key.lower() for keyword in ["end", "to", "结束"]):
                    end_date_param = key

        # 对于历史数据API，自动计算当前日期往前10个交易日的日期范围
        if is_historical_api:
            # 准备基本参数（除日期外的其他参数）
            base_params = {}
            if params:
                for key, value in params.items():
                    if (start_date_param and key == start_date_param) or (end_date_param and key == end_date_param):
                        # 日期参数跳过，将使用自动计算的日期范围
                        continue
                    elif isinstance(value, str):
                        if any(date_keyword in key.lower() for date_keyword in ["date", "time", "日期", "时间"]):
                            base_params[key] = f"format_date_param('{value}')"
                        else:
                            base_params[key] = f"'{value}'"
                    else:
                        base_params[key] = str(value)

            # 构建基本参数字符串
            base_param_strs = [f"{key}={value}" for key, value in base_params.items()]
            base_param_str = ", ".join(base_param_strs)
            if base_param_str:
                base_param_str += ", "

            # 自动计算日期范围的代码
            code_lines.extend([
                "",
                "# 自动计算日期范围（当前日期往前10个交易日）",
                "end_date = datetime.now()",
                "start_date = end_date - timedelta(days=20)  # 往前20天，确保能覆盖10个交易日",
                "",
                "# 转换为akshare所需的格式",
                "end_date_str = end_date.strftime('%Y%m%d')",
                "start_date_str = start_date.strftime('%Y%m%d')",
                "",

            ])

            # 生成API调用代码
            if start_date_param and end_date_param:
                code_lines.append(f"result = ak.{api_name}({base_param_str}{start_date_param}=start_date_str, {end_date_param}=end_date_str)")
            else:
                # 如果未明确识别出日期参数名称，但仍是历史API，使用默认命名
                code_lines.append(f"# 注意：未明确识别出日期参数名称，使用通用参数名")
                code_lines.append(f"result = ak.{api_name}({base_param_str}start_date=start_date_str, end_date=end_date_str)")
        else:
            # 标准API调用（非历史数据API）
            if params:
                # 格式化参数，对日期参数进行特殊处理
                param_strs = []
                for key, value in params.items():
                    # 识别日期参数并格式化
                    param_is_date = (
                        key in date_params or
                        any(date_keyword in key.lower() for date_keyword in ["date", "time", "start", "end", "日期", "时间", "开始", "结束"])
                    )

                    if param_is_date and isinstance(value, str):
                        param_strs.append(f"{key}=format_date_param('{value}')")
                    elif isinstance(value, str):
                        param_strs.append(f"{key}='{value}'")
                    else:
                        param_strs.append(f"{key}={value}")

                param_str = ", ".join(param_strs)
                code_lines.append(f"result = ak.{api_name}({param_str})")
            else:
                code_lines.append(f"result = ak.{api_name}()")

        code_lines.extend([
            "",
            "# Process and format the result",
            "if isinstance(result, pd.DataFrame):",
            "    if len(result) > 0:",
            "        # Convert DataFrame to dict for JSON serialization",
            "        result_dict = result.head(20).to_dict(orient='records')",
            "        summary = {",
            "            'total_records': len(result),",
            "            'columns': list(result.columns),",
            "            'data_sample': result_dict",
            "        }",
            "",
            "        # Add basic statistics for numeric columns",
            "        numeric_stats = {}",
            "        for col in result.select_dtypes(include=['number']).columns:",
            "            numeric_stats[col] = {",
            "                'mean': float(result[col].mean()),",
            "                'min': float(result[col].min()),",
            "                'max': float(result[col].max())",
            "            }",
            "        ",
            "        if numeric_stats:",
            "            summary['statistics'] = numeric_stats",
        ])

        if plot:
            code_lines.extend([
                "",
                "        # Generate plot",
                "        if 'date' in result.columns or '日期' in result.columns:",
                "            date_col = 'date' if 'date' in result.columns else '日期'",
                "            # Check for price columns",
                "            price_cols = [col for col in result.columns if col in ['close', '收盘', '收盘价']]",
                "            volume_cols = [col for col in result.columns if col in ['volume', '成交量']]",
                "            ",
                "            if price_cols and volume_cols:",
                "                # Create subplots",
                "                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,",
                "                                  vertical_spacing=0.1,",
                "                                  subplot_titles=('Price', 'Volume'),",
                "                                  row_heights=[0.7, 0.3])",
                "                ",
                "                # Add price line",
                "                fig.add_trace(",
                "                    go.Scatter(",
                "                        x=result[date_col],",
                "                        y=result[price_cols[0]],",
                "                        name=price_cols[0]",
                "                    ),",
                "                    row=1, col=1",
                "                )",
                "                ",
                "                # Add volume bar",
                "                fig.add_trace(",
                "                    go.Bar(",
                "                        x=result[date_col],",
                "                        y=result[volume_cols[0]],",
                "                        name=volume_cols[0]",
                "                    ),",
                "                    row=2, col=1",
                "                )",
                "                ",
                "                # Update layout",
                "                fig.update_layout(",
                f"                    title='{api_name} Data Visualization',",
                "                    xaxis_title='Date',",
                "                    yaxis_title='Price',",
                "                    height=800,",
                "                    width=1200",
                "                )",
                "                ",
                "                # Save plot to temp file",
                "                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp:",
                "                    temp_path = temp.name",
                "                    fig.write_html(temp_path)",
                "                ",
                "                summary['plot_path'] = temp_path",
            ])

        code_lines.extend([
            "",
            "        output = json.dumps(summary, ensure_ascii=False, indent=2, cls=DateTimeEncoder)",
            "    else:",
            "        output = json.dumps({'message': 'DataFrame is empty'}, ensure_ascii=False)",
            "else:",
            "    output = json.dumps({'data': str(result)}, ensure_ascii=False)",
            "",
            "print(output)"
        ])

        return "\n".join(code_lines)

    def _execute_generated_code(self, code: str) -> Dict:
        """
        执行生成的代码并捕获输出。
        同时将生成的代码和执行结果保存到工作空间。

        参数:
            code: 要执行的Python代码

        返回:
            包含执行结果的字典
        """
        try:
            # 生成唯一ID用于代码文件和结果文件
            execution_id = generate_timestamp_id("stock_code")

            # 保存代码到工作空间
            code_filename = f"{execution_id}.py"
            code_path = save_to_workspace(code, code_filename, "stock_code")

            logger.info(f"数据获取代码已保存到: {code_path}")

            # 执行代码并捕获输出
            import subprocess
            result = subprocess.run(['python', code_path], capture_output=True, text=True)

            # 打印执行结果
            logger.info(f"代码执行结果: {result.stdout}")
            if result.stderr:
                logger.warning(f"代码执行警告/错误: {result.stderr}")
            if result.returncode != 0:
                error_info = {
                    "status": "error",
                    "error": result.stderr,
                    "message": "代码执行失败",
                    "code_file": code_path
                }
                # 保存错误信息
                save_to_workspace(error_info, f"{execution_id}_error.json", "stock_code", is_json=True)
                return error_info

            # 尝试将输出解析为JSON
            try:
                output_data = json.loads(result.stdout)
                output_data["status"] = "success"
                output_data["code_file"] = code_path

                # 如果输出包含数据，也将数据保存到工作空间
                if "data" in output_data:
                    data_content = output_data["data"]
                    # 保存数据（可能是字符串表示的数据帧或其他数据）
                    data_file = f"{execution_id}_data.json"
                    data_path = save_to_workspace(data_content, data_file, "stock_data", is_json=True)
                    output_data["data_file"] = data_path

                # 保存成功的结果
                results_path = save_to_workspace(output_data, f"{execution_id}_result.json", "stock_code", is_json=True)
                output_data["results_file"] = results_path

                return output_data
            except json.JSONDecodeError:
                # 对于非JSON输出，直接保存原始输出
                output_info = {
                    "status": "success",
                    "output_text": result.stdout,
                    "message": "输出不是JSON格式",
                    "code_file": code_path
                }
                # 尝试将原始输出转换为规范的JSON格式
                try:
                    # 检查输出是否是表格格式
                    lines = result.stdout.strip().split('\n')

                    if lines and ('\t' in lines[0] or ',' in lines[0]):
                        # 尝试将表格数据转换为CSV格式
                        try:
                            import pandas as pd
                            import io

                            # 根据分隔符创建DataFrame
                            delimiter = '\t' if '\t' in lines[0] else ','
                            df = pd.read_csv(io.StringIO(result.stdout), sep=delimiter)

                            # 生成CSV文件名
                            csv_filename = f"{execution_id}_output.csv"
                            # 保存为CSV文件
                            output_path = save_to_workspace(df.to_csv(index=False), csv_filename, "stock_code")

                            # 创建CSV输出的元数据
                            csv_metadata = {
                                "file_path": output_path,
                                "columns": list(df.columns),
                                "total_rows": len(df),
                                "format": "csv"
                            }

                            # 保存元数据
                            metadata_path = save_to_workspace(csv_metadata, f"{execution_id}_metadata.json", "stock_code", is_json=True)
                            output_path = metadata_path

                        except Exception as e:
                            # 如果转换失败，记录原始输出
                            logger.exception(f"转换表格数据为CSV时出错: {str(e)}")
                            error_data = {
                                "raw_data": result.stdout,
                                "message": "无法解析为CSV格式，提供原始输出",
                                "timestamp": datetime.now().isoformat()
                            }
                            output_path = save_to_workspace(error_data, f"{execution_id}_error.json", "stock_code", is_json=True)
                    else:
                        # 非表格数据，保存原始文本
                        raw_data = {
                            "raw_data": result.stdout,
                            "message": "非表格格式数据，无法转换为CSV",
                            "timestamp": datetime.now().isoformat()
                        }
                        output_path = save_to_workspace(raw_data, f"{execution_id}_raw.json", "stock_code", is_json=True)

                except Exception as e:
                    logger.exception(f"转换为CSV格式时出错: {str(e)}")
                    # 保存错误信息
                    error_data = {
                        "raw_data": result.stdout,
                        "message": "处理输出时出错，提供原始数据",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    output_path = save_to_workspace(default_json, f"{execution_id}_output.json", "stock_code", is_json=True)

                output_info["output_file"] = output_path

                # 保存结果信息
                save_to_workspace(output_info, f"{execution_id}_result.json", "stock_code", is_json=True)

                return output_info

        except Exception as e:
            logger.exception("代码执行过程中出错")
            error_result = {
                "status": "error",
                "error": str(e),
                "message": "代码执行过程中出错"
            }
            # 尝试保存错误信息和代码
            try:
                error_id = generate_timestamp_id("stock_error")
                # 保存代码
                code_path = save_to_workspace(code, f"{error_id}.py", "stock_code")
                error_result["code_file"] = code_path
                # 保存错误信息
                save_to_workspace(error_result, f"{error_id}_error.json", "stock_code", is_json=True)
            except:
                logger.exception("保存错误信息时出错")

            return error_result


# 用于测试
if __name__ == "__main__":
    import asyncio

    async def test():
        # 测试API分析器
        analyzer = StockAPIAnalyzerTool()
        result = await analyzer.execute(
            query="我需要获取A股市场中贵州茅台的历史股价数据"
        )
        print("分析器结果:", result)

        # 测试代码生成器
        generator = StockCodeGeneratorTool()
        params = {
            "api_name": "stock_zh_a_hist",
            "params": {
                "symbol": "600519",
                "period": "daily",
                "start_date": "20230101",
                "end_date": "20230131"
            },
            "execute": False
        }
        result = await generator.execute(json.dumps(params))
        print("生成器结果:", result)

    asyncio.run(test())
