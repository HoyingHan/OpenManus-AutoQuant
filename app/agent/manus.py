from typing import Optional
import os

from pydantic import Field, model_validator

from app.agent.browser import BrowserContextHelper
from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.python_execute import PythonExecute
from app.tool.str_replace_editor import StrReplaceEditor
from app.tool.stock_data_fetch import StockAPIAnalyzerTool, StockCodeGeneratorTool
from app.tool.stock_analysis import StockAnalysisTool
from app.tool.strategy_generator import StrategyGeneratorTool
from app.tool.backtest import BacktestTool
from app.tool.strategy_optimizer import StrategyOptimizerTool


class Manus(ToolCallAgent):
    """A versatile general-purpose agent with quantitative analysis capabilities."""

    name: str = "Manus"
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools, "
        "including quantitative analysis and trading strategy optimization"
    )

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 20

    # Add general-purpose tools and quant tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            BrowserUseTool(),
            StrReplaceEditor(),
            Terminate(),
            StockAPIAnalyzerTool(),
            StockCodeGeneratorTool(),
            StockAnalysisTool(),
            StrategyGeneratorTool(),
            BacktestTool(),
            StrategyOptimizerTool()
        )
    )

    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])

    browser_context_helper: Optional[BrowserContextHelper] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 创建量化分析所需的工作目录
        workspace_dirs = [
            "workspace",
            "workspace/data",
            "workspace/strategies",
            "workspace/backtest_results",
            "workspace/optimization_results",
            "workspace/analysis_reports"
        ]
        for dir_path in workspace_dirs:
            os.makedirs(os.path.join(os.getcwd(), dir_path), exist_ok=True)

    @model_validator(mode="after")
    def initialize_helper(self) -> "Manus":
        self.browser_context_helper = BrowserContextHelper(self)
        return self

    async def think(self) -> bool:
        """Process current state and decide next actions with appropriate context."""
        original_prompt = self.next_step_prompt
        recent_messages = self.memory.messages[-3:] if self.memory.messages else []
        browser_in_use = any(
            tc.function.name == BrowserUseTool().name
            for msg in recent_messages
            if msg.tool_calls
            for tc in msg.tool_calls
        )

        if browser_in_use:
            self.next_step_prompt = (
                await self.browser_context_helper.format_next_step_prompt()
            )

        result = await super().think()

        # Restore original prompt
        self.next_step_prompt = original_prompt

        return result

    async def cleanup(self):
        """Clean up Manus agent resources."""
        if self.browser_context_helper:
            await self.browser_context_helper.cleanup_browser()
