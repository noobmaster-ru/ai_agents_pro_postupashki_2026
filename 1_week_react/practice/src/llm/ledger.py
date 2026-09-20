from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Ledger:
    calls: list = field(default_factory=list)

    def add(self, tag: str, model: str, usage: dict, seconds: float = 0.0, run_id: str | None = None) -> float:
        cost = usage.get("cost") or 0.0
        self.calls.append({"tag": tag, "model": model.split("/")[-1],
                           "prompt": usage.get("prompt_tokens", 0), "completion": usage.get("completion_tokens", 0),
                           "cost": cost, "seconds": round(seconds, 2), "run_id": run_id})
        return cost

    @property
    def total(self) -> float:
        return sum(c["cost"] for c in self.calls)

    def by_tag(self, tag: str) -> list:
        return [c for c in self.calls if c["tag"] == tag]

    def by_run(self, run_id: str) -> list:
        return [c for c in self.calls if c["run_id"] == run_id]

    def cost_of(self, run_id: str) -> float:
        return sum(c["cost"] for c in self.by_run(run_id))

    def table(self) -> pd.DataFrame:
        df = pd.DataFrame(self.calls)
        return df.groupby(["tag", "model"]).agg(calls=("cost", "size"), prompt=("prompt", "sum"),
                                                completion=("completion", "sum"), cost=("cost", "sum"),
                                                seconds=("seconds", "sum")).round(5)
