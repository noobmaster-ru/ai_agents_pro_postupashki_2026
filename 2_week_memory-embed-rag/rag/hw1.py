import sys
from pathlib import Path

WEEK1 = Path(__file__).resolve().parents[2] / "1_week_react" / "practice"
if str(WEEK1) not in sys.path:
    sys.path.insert(0, str(WEEK1))

from src.config import Settings, MODELS, COLORS  # noqa: E402
from src.llm import Ledger, LLMClient, Answer, ask_structured, json_from  # noqa: E402
from src.tools import ToolRegistry, Tool  # noqa: E402
from src.agent import Agent, Run, show_trace  # noqa: E402
from src.evaluation import load_tasks, normalize, is_correct, final_answer  # noqa: E402

__all__ = ["WEEK1", "Settings", "MODELS", "COLORS", "Ledger", "LLMClient", "Answer", "ask_structured", "json_from",
           "ToolRegistry", "Tool", "Agent", "Run", "show_trace", "load_tasks", "normalize", "is_correct", "final_answer"]
