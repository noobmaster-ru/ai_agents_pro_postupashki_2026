import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm.auto import tqdm

from ..agent.agent import Agent, Run
from ..llm.client import LLMClient
from ..llm.structured import ask_structured
from ..tools.registry import ToolRegistry
from .tasks import final_answer, is_correct

CONFIGS = {"без инструментов": [], "калькулятор": ["calculator"], "код": ["python_exec"], "поиск": ["web_search"],
           "поиск + страница": ["web_search", "page_find"],
           "все инструменты": ["web_search", "page_find", "calculator", "python_exec"],
           "безымянные": ["tool_a", "tool_b"]}


def short(model: str) -> str:
    return model.split("/")[-1]


def summary(config: str, model: str, n: int, correct: int, cost: float, steps: float, seconds: float = 0.0) -> dict:
    return {"config": config, "model": short(model), "n": n, "accuracy": round(correct / n, 2),
            "cost_per_task": round(cost / n, 5), "cost_per_correct": round(cost / correct, 5) if correct else float("inf"),
            "avg_steps": round(steps, 2), "avg_seconds": round(seconds, 1)}


def report(results: pd.DataFrame) -> pd.DataFrame:
    rows = [summary(config, model, len(df), int(df["correct"].sum()), df["cost"].sum(), df["steps"].mean(), df["seconds"].mean())
            for (config, model), df in results.groupby(["config", "model"])]
    return pd.DataFrame(rows).sort_values("cost_per_task").reset_index(drop=True)


class Experiment:
    def __init__(self, client: LLMClient, registry: ToolRegistry, traces_dir: Path, workers: int = 1):
        self.client = client
        self.registry = registry
        self.traces_dir = Path(traces_dir)
        self.workers = workers

    def run_tasks(self, tasks: list[dict], model: str, tool_names: list[str], config: str, max_steps: int = 8,
                  on_row: Callable[[dict], None] | None = None) -> pd.DataFrame:
        agent = Agent(self.client, self.registry, model, tool_names, max_steps)
        folder = self.traces_dir / config / short(model)
        folder.mkdir(parents=True, exist_ok=True)
        rows = {}
        bar = tqdm(total=len(tasks), desc=f"{config} | {short(model)}", unit="задача", leave=True)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self._safe_run, agent, task): task for task in tasks}
            for future in as_completed(futures):
                task, run = futures[future], future.result()
                ok = is_correct(task, final_answer(run.answer))
                rows[task["id"]] = self._row(task, run, ok, config, model)
                self._save_trace(folder / f"{task['id']}.json", task, run, ok, config, model)
                if on_row:
                    on_row(rows[task["id"]])
                bar.update(1)
                bar.set_postfix(self._progress(rows.values()))
        bar.close()
        return pd.DataFrame([rows[t["id"]] for t in tasks])

    def compare_models(self, tasks: list[dict], models: dict[str, str], config: str = "без инструментов, JSON") -> pd.DataFrame:
        rows = []
        for model in models.values():
            before, correct, started = self.client.ledger.total, 0, time.perf_counter()
            for task in tqdm(tasks, desc=f"{config} | {short(model)}", unit="задача"):
                try:
                    correct += is_correct(task, ask_structured(self.client, task["question"], model).final)
                except Exception:
                    pass
            rows.append(summary(config, model, len(tasks), correct, self.client.ledger.total - before, 1,
                                (time.perf_counter() - started) / len(tasks)))
        return pd.DataFrame(rows)

    @staticmethod
    def _progress(rows) -> dict:
        rows = list(rows)
        correct, cost = sum(r["correct"] for r in rows), sum(r["cost"] for r in rows)
        return {"верно": f"{correct / len(rows):.0%}", "цена": f"${cost:.3f}", "шагов": f"{sum(r['steps'] for r in rows) / len(rows):.1f}"}

    @staticmethod
    def _safe_run(agent: Agent, task: dict) -> Run:
        try:
            return agent.run(task["question"])
        except Exception as e:
            return Run(task["question"], f"ошибка: {e}", 0, [])

    @staticmethod
    def _row(task: dict, run: Run, ok: bool, config: str, model: str) -> dict:
        return {"config": config, "model": short(model), "id": task["id"], "source": task["source"],
                "correct": ok, "steps": run.steps, "tool_calls": run.tool_calls, "cost": run.cost,
                "seconds": round(run.seconds, 1), "answer": final_answer(run.answer)[:60], "gold": task["answer"]}

    @staticmethod
    def _save_trace(path: Path, task: dict, run: Run, ok: bool, config: str, model: str) -> None:
        trace = {"task": task, "config": config, "model": short(model), "answer": run.answer,
                 "final": final_answer(run.answer), "correct": ok, "steps": run.steps, "cost": run.cost,
                 "seconds": round(run.seconds, 1), "messages": run.messages}
        path.write_text(json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8")
