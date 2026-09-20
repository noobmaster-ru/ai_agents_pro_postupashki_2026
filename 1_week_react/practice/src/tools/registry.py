from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel

RESULT_LIMIT = 2000


@dataclass(frozen=True)
class Tool:
    fn: Callable[..., str]
    args: type[BaseModel]
    description: str

    @property
    def name(self) -> str:
        return self.fn.__name__

    @property
    def schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.args.model_json_schema()}}

    def run(self, arguments: str) -> str:
        args = self.args.model_validate_json(arguments)
        return str(self.fn(**args.model_dump()))


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, fn: Callable[..., str], args_model: type[BaseModel], description: str) -> Tool:
        tool = Tool(fn, args_model, description)
        self._tools[tool.name] = tool
        return tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __getitem__(self, name: str) -> Tool:
        return self._tools[name]

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self, names: list[str]) -> list[dict]:
        return [self._tools[n].schema for n in names]

    def call(self, tool_call: dict) -> dict:
        name = tool_call["function"]["name"]
        try:
            result = self._tools[name].run(tool_call["function"]["arguments"])
        except Exception as e:
            result = f"не удалось вызвать инструмент {name}: {e}"
        return {"role": "tool", "tool_call_id": tool_call["id"], "content": result[:RESULT_LIMIT]}
