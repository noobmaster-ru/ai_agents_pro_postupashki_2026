import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import pandas as pd

from .hw1 import is_correct
from .retrieval import is_gold


def final_answer(text: str) -> str:
    found = re.findall(r"FINAL:\s*(.+)", text or "")
    return found[-1].strip() if found else (text or "").strip()


def is_refusal(text: str) -> bool:
    return "NOT_FOUND" in final_answer(text)


def is_right(task: dict, answer: str) -> bool:
    return bool(task.get("answer")) and is_correct(task, final_answer(answer))


def evaluate(fn: Callable[[str, str], dict], tasks: list[dict], config: str, model: str, workers: int = 6,
             cache: Path | None = None, traces: list | None = None) -> pd.DataFrame:
    if cache and Path(cache).exists():
        return pd.read_csv(cache)

    def one(task: dict) -> dict:
        try:
            out = fn(task["question"], model)
        except Exception as e:
            out = {"answer": f"ошибка: {e}", "hits": [], "cost": 0.0, "steps": 0, "searches": 0, "messages": []}
        if traces is not None:
            traces.append({"config": config, "model": model, "task": task, **{k: v for k, v in out.items() if k != "hits"},
                           "hits": [{k: h.get(k) for k in ("id", "page", "section", "score", "text")} for h in out["hits"]]})
        return {"config": config, "model": model.split("/")[-1], "id": task["id"], "correct": is_right(task, out["answer"]),
                "found": any(is_gold(h, task) for h in out["hits"]) if task.get("evidence") else False,
                "refused": is_refusal(out["answer"]), "cost": out["cost"], "steps": out.get("steps", 1),
                "searches": out.get("searches", 0), "answer": final_answer(out["answer"])[:80], "gold": task.get("answer", "")}

    with ThreadPoolExecutor(workers) as pool:
        frame = pd.DataFrame(list(pool.map(one, tasks)))
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache, index=False)
    return frame


def report(results: pd.DataFrame) -> pd.DataFrame:
    g = results.groupby(["config", "model"], sort=False)
    table = g.agg(n=("correct", "size"), accuracy=("correct", "mean"), cost_per_task=("cost", "mean"),
                  avg_searches=("searches", "mean")).reset_index()
    table["cost_per_correct"] = table["cost_per_task"] / table["accuracy"].replace(0, float("nan"))
    return table.round(5)


def refusal_table(answerable: pd.DataFrame, unanswerable: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (config, model), part in answerable.groupby(["config", "model"], sort=False):
        none = unanswerable[(unanswerable["config"] == config) & (unanswerable["model"] == model)]
        rows.append({"config": config, "model": model,
                     "отклонено без ответа": f"{int(none['refused'].sum())} из {len(none)}",
                     "отклонено зря": f"{int(part['refused'].sum())} из {len(part)}",
                     "доля отказов зря": round(float(part["refused"].mean()), 3)})
    return pd.DataFrame(rows)


def diagnose(results: pd.DataFrame, config: str, model: str) -> dict:
    part = results[(results["config"] == config) & (results["model"] == model)]
    wrong = part[~part["correct"]]
    return {"всего провалов": int(len(wrong)), "поиск не нашёл": int((~wrong["found"]).sum()),
            "нашёл, но ответ неверный": int(wrong["found"].sum()),
            "ids не нашёл": list(wrong[~wrong["found"]]["id"]), "ids нашёл": list(wrong[wrong["found"]]["id"])}


def last_week_rows(tasks: list[dict], week1_csv: Path, config: str, model: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(week1_csv)
    df = df[(df["config"] == config) & (df["model"] == model)].set_index("id")
    ids = [t["id"] for t in tasks if t["id"] in df.index]
    part = df.loc[ids]
    return pd.DataFrame({"config": label, "model": model, "id": ids, "correct": part["correct"].astype(bool).values,
                         "found": None, "refused": False, "cost": part["cost"].values, "steps": part["steps"].values,
                         "searches": part["tool_calls"].values, "answer": part["answer"].values, "gold": part["gold"].values})
