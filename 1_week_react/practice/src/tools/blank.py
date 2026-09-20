from pydantic import BaseModel

from .calculator import calculator
from .python_exec import python_exec
from .registry import ToolRegistry


class BlankArgs(BaseModel):
    text: str


def tool_a(text: str) -> str:
    return calculator(text)


def tool_b(text: str) -> str:
    return python_exec(text)


def register_blank_tools(registry: ToolRegistry) -> None:
    registry.register(tool_a, BlankArgs, "Инструмент")
    registry.register(tool_b, BlankArgs, "Инструмент")
