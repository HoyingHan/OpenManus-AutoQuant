import asyncio
import json
from app.tool.stock_data_fetch import StockAPIAnalyzerTool, StockCodeGeneratorTool

async def test_api_analyzer():
    """测试API分析工具"""
    print("==== API分析工具测试结果 ====")
    analyzer = StockAPIAnalyzerTool()
    result = await analyzer.execute("我需要获取A股市场中贵州茅台的历史股价数据")
    # 处理ToolResult对象
    if hasattr(result, 'output') and result.output is not None:
        print(result.output)
    elif hasattr(result, 'error') and result.error is not None:
        print(f"错误: {result.error}")
    else:
        print("无结果")
    print("\n")

async def test_code_generator_only():
    """测试代码生成功能，但不执行生成的代码"""
    print("==== 代码生成功能测试结果 ====")
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
    # 处理ToolResult对象
    if hasattr(result, 'output') and result.output is not None:
        try:
            print(json.dumps(json.loads(result.output), ensure_ascii=False, indent=2))
        except:
            print(result.output)
    elif hasattr(result, 'error') and result.error is not None:
        print(f"错误: {result.error}")
    else:
        print("无结果")
    print("\n")

async def test_code_generator_execute():
    """测试代码生成和执行功能"""
    print("==== 代码生成和执行功能测试结果 ====")
    generator = StockCodeGeneratorTool()
    params = {
        "api_name": "stock_zh_a_hist",
        "params": {
            "symbol": "600519",
            "period": "daily",
            "start_date": "20230101",
            "end_date": "20230131"
        },
        "execute": True
    }
    result = await generator.execute(json.dumps(params))
    # 处理ToolResult对象
    if hasattr(result, 'output') and result.output is not None:
        try:
            print(json.dumps(json.loads(result.output), ensure_ascii=False, indent=2))
        except:
            print(result.output)
    elif hasattr(result, 'error') and result.error is not None:
        print(f"错误: {result.error}")
    else:
        print("无结果")
    print("\n")

async def main():
    await test_api_analyzer()
    await test_code_generator_only()
    await test_code_generator_execute()

if __name__ == "__main__":
    asyncio.run(main())
