import uuid

import pandas as pd
from tqdm.auto import tqdm

from ..llm.client import LLMClient
from ..llm.schemas import Confident
from ..llm.structured import ask_structured
from .tasks import is_correct


class Cascade:
    def __init__(self, client: LLMClient, cheap: str, strong: str):
        self.client = client
        self.cheap = cheap
        self.strong = strong
        self._memo: dict = {}

    def confident_answer(self, task: dict, model: str) -> dict:
        key = (task["id"], model)
        if key not in self._memo:
            run_id = uuid.uuid4().hex
            try:
                ans = ask_structured(self.client, task["question"], model, Confident, run_id=run_id)
                result = {"final": ans.final, "confidence": ans.confidence, "correct": is_correct(task, ans.final)}
            except Exception:
                result = {"final": "", "confidence": 0.0, "correct": False}
            self._memo[key] = {**result, "cost": self.client.ledger.cost_of(run_id)}
        return dict(self._memo[key])

    def answer(self, task: dict, threshold: float) -> dict:
        cheap = self.confident_answer(task, self.cheap)
        if cheap["confidence"] >= threshold:
            return {**cheap, "escalated": False}
        strong = self.confident_answer(task, self.strong)
        return {**strong, "cost": cheap["cost"] + strong["cost"], "escalated": True}

    def sweep(self, tasks: list[dict], thresholds: list[float]) -> pd.DataFrame:
        rows = []
        for th in thresholds:
            outs = [self.answer(t, th) for t in tqdm(tasks, desc=f"каскад, порог {th}", unit="задача")]
            rows.append({"threshold": th, "accuracy": sum(o["correct"] for o in outs) / len(outs),
                         "cost_per_task": sum(o["cost"] for o in outs) / len(outs),
                         "escalated": sum(o["escalated"] for o in outs) / len(outs)})
        df = pd.DataFrame(rows)
        df["cost_per_correct"] = df["cost_per_task"] / df["accuracy"].replace(0, float("nan"))
        return df.round(5)

    def confidence_table(self, tasks: list[dict], model: str) -> pd.DataFrame:
        return pd.DataFrame([self.confident_answer(t, model) for t in tasks])
