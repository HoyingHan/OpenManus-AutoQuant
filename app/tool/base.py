from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import os
import json
from pathlib import Path
from datetime import datetime  # 导入datetime类

from pydantic import BaseModel, Field


def ensure_workspace_dir(subdir: str = None) -> str:
    """
    确保工作空间目录存在，并返回目录路径。

    参数:
        subdir: 工作空间内的子目录名称，如果不提供则返回工作空间根目录

    返回:
        工作空间目录的完整路径
    """
    # 工作空间根目录
    workspace_dir = os.path.join(os.getcwd(), "workspace")

    # 如果没有指定子目录，确保根目录存在后返回
    if not subdir:
        os.makedirs(workspace_dir, exist_ok=True)
        return workspace_dir

    # 确保子目录存在
    target_dir = os.path.join(workspace_dir, subdir)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


def save_to_workspace(content: Any, filename: str, subdir: str = None, is_json: bool = False) -> str:
    """
    将内容保存到工作空间目录。

    参数:
        content: 要保存的内容
        filename: 文件名
        subdir: 工作空间内的子目录名称
        is_json: 内容是否为JSON对象，如果是则格式化保存

    返回:
        保存文件的完整路径
    """
    from app.logger import logger

    # 确保目录存在
    target_dir = ensure_workspace_dir(subdir)

    # 构建完整的文件路径
    file_path = os.path.join(target_dir, filename)

    # 保存内容
    try:
        if is_json:
            with open(file_path, 'w', encoding='utf-8') as f:
                if isinstance(content, (dict, list)):
                    json.dump(content, f, ensure_ascii=False, indent=2)
                else:
                    # 如果内容是JSON字符串
                    f.write(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2))
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

        logger.info(f"文件已保存到工作空间: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"保存文件到工作空间时出错: {str(e)}")
        return None


def generate_timestamp_id(prefix: str = "") -> str:
    """
    生成带有时间戳的唯一ID。

    参数:
        prefix: ID前缀

    返回:
        格式为 "prefix_YYYYMMDDHHmmss" 的ID字符串
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{prefix}_{timestamp}" if prefix else timestamp


class BaseTool(ABC, BaseModel):
    """
    工具基类，所有特定功能工具都应该继承这个类。
    提供了工具执行的基本框架和日志记录功能。
    """
    name: str  # 工具名称，用于标识和调用
    description: str  # 工具描述，用于帮助用户理解工具功能
    parameters: Optional[dict] = None  # 工具参数定义，用于验证输入

    class Config:
        arbitrary_types_allowed = True  # 允许任意类型，便于扩展

    async def __call__(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        from app.logger import logger

        # 记录工具执行的输入参数
        logger.info(f"执行工具: {self.name}")
        logger.info(f"工具输入参数: {kwargs}")

        # 执行工具
        result = await self.execute(**kwargs)

        # 记录工具执行的结果
        if isinstance(result, ToolResult):
            if result.error:
                logger.error(f"工具执行错误: {result.error}")
            else:
                logger.info(f"工具执行结果: {result.output}")
        else:
            logger.info(f"工具执行结果: {result}")

        return result

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""

    def to_param(self) -> Dict:
        """Convert tool to function call format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolResult(BaseModel):
    """Represents the result of a tool execution."""

    output: Any = Field(default=None)
    error: Optional[str] = Field(default=None)
    base64_image: Optional[str] = Field(default=None)
    system: Optional[str] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        """Initialize a tool result with logging."""
        from app.logger import logger

        super().__init__(**data)

        # 记录工具结果的创建
        if self.error:
            logger.warning(f"创建了带有错误的工具结果: {self.error}")
        elif self.output:
            logger.debug(f"创建了工具结果: {str(self.output)[:100]}{'...' if len(str(self.output)) > 100 else ''}")

    def __bool__(self):
        return any(getattr(self, field) for field in self.__fields__)

    def __add__(self, other: "ToolResult"):
        from app.logger import logger

        logger.debug(f"合并工具结果: {type(self).__name__} + {type(other).__name__}")

        def combine_fields(
            field: Optional[str], other_field: Optional[str], concatenate: bool = True
        ):
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("Cannot combine tool results")
            return field or other_field

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            base64_image=combine_fields(self.base64_image, other.base64_image, False),
            system=combine_fields(self.system, other.system),
        )

    def __str__(self):
        from app.logger import logger
        result = f"Error: {self.error}" if self.error else self.output
        logger.debug(f"工具结果转换为字符串: {result if isinstance(result, str) else type(result)}")
        return result

    def replace(self, **kwargs):
        """Returns a new ToolResult with the given fields replaced."""
        from app.logger import logger
        logger.debug(f"替换工具结果字段: {kwargs.keys()}")
        # return self.copy(update=kwargs)
        return type(self)(**{**self.dict(), **kwargs})


class CLIResult(ToolResult):
    """A ToolResult that can be rendered as a CLI output."""


class ToolFailure(ToolResult):
    """A ToolResult that represents a failure."""
