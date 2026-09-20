from pathlib import Path

from .config import Settings, MODELS, COLORS
from .llm import Ledger, LLMClient, Answer, Numeric, Confident, ask_structured, json_from
from .tools import ToolRegistry, default_registry, register_blank_tools
from .agent import Agent, Run, show_trace
from .evaluation import Experiment, Cascade, CONFIGS, load_tasks, is_correct, final_answer, report, summary


def bootstrap(data_dir: Path | None = None) -> tuple[Settings, Ledger, LLMClient, ToolRegistry]:
    settings = Settings.from_env(data_dir)
    ledger = Ledger()
    client = LLMClient(settings, ledger)
    registry = default_registry()
    register_blank_tools(registry)
    return settings, ledger, client, registry


__all__ = ["Settings", "MODELS", "COLORS", "Ledger", "LLMClient", "Answer", "Numeric", "Confident", "ask_structured",
           "json_from", "ToolRegistry", "default_registry", "register_blank_tools", "Agent", "Run", "show_trace",
           "Experiment", "Cascade", "CONFIGS", "load_tasks", "is_correct", "final_answer", "report", "summary", "bootstrap"]
