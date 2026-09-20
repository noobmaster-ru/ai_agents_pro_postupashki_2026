from .registry import Tool, ToolRegistry
from .calculator import calculator, CalcArgs
from .wiki import web_search, page_find, SearchArgs, PageFindArgs
from .python_exec import python_exec, ExecArgs
from .blank import register_blank_tools


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(calculator, CalcArgs, "Считает арифметическое выражение: числа, скобки, + - * / ** %")
    registry.register(web_search, SearchArgs, "Ищет статью в англоязычной Википедии и возвращает её вступление")
    registry.register(page_find, PageFindArgs,
                      "Берёт полный текст статьи англоязычной Википедии по точному названию и возвращает только "
                      "строки, где встречаются ключевые слова. Нужен, когда вступления статьи не хватает")
    registry.register(python_exec, ExecArgs, "Выполняет код на Python в отдельном процессе и возвращает то, что он напечатал")
    return registry


__all__ = ["Tool", "ToolRegistry", "default_registry", "register_blank_tools",
           "calculator", "web_search", "page_find", "python_exec",
           "CalcArgs", "SearchArgs", "PageFindArgs", "ExecArgs"]
