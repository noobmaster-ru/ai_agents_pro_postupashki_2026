import argparse
import json
import sys
import threading
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import bootstrap, MODELS, Experiment, CONFIGS, load_tasks, report
from src.viz import money_chart

PLAN = [("strong", "без инструментов"), ("cheap", "без инструментов"), ("cheap", "поиск"),
        ("cheap", "поиск + страница"), ("mid", "поиск + страница")]


def pending_tasks(done: pd.DataFrame, tasks: list[dict], config: str, model: str) -> list[dict]:
    if done.empty:
        return tasks
    rows = done[(done["config"] == config) & (done["model"] == model.split("/")[-1]) & (done["steps"] > 0)]
    finished = set(rows["id"])
    return [t for t in tasks if t["id"] not in finished]


def merge(done: pd.DataFrame, fresh: pd.DataFrame, config: str, model: str) -> pd.DataFrame:
    if done.empty:
        return fresh
    keep = ~((done["config"] == config) & (done["model"] == model.split("/")[-1]) & (done["id"].isin(fresh["id"])))
    return pd.concat([done[keep], fresh], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fresh.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--out", default="results")
    ap.add_argument("--traces", default="traces")
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--plan", default="", help="cheap:поиск,mid:поиск + страница; пусто = весь план")
    ap.add_argument("--suffix", default="", help="добавка к имени конфигурации, например ' v2'")
    args = ap.parse_args()
    plan = [tuple(item.split(":", 1)) for item in args.plan.split(",") if item.strip()] or PLAN

    settings, ledger, client, registry = bootstrap()
    tasks = load_tasks(args.data)
    if args.limit:
        tasks = tasks[:args.limit]
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    raw_path = out / "results_raw.csv"
    results = pd.read_csv(raw_path) if args.resume and raw_path.exists() else pd.DataFrame()
    exp = Experiment(client, registry, Path(args.traces), workers=args.workers)

    for model_name, base_config in plan:
        model, config = MODELS[model_name], base_config + args.suffix
        todo = pending_tasks(results, tasks, config, model)
        if todo:
            lock = threading.Lock()

            def flush(row):
                nonlocal results
                with lock:
                    results = merge(results, pd.DataFrame([row]), config, model)
                    results.to_csv(raw_path, index=False)

            df = exp.run_tasks(todo, model, CONFIGS[base_config], config, max_steps=args.max_steps, on_row=flush)
            results = merge(results, df, config, model)
            results.to_csv(raw_path, index=False)
        part = results[(results["config"] == config) & (results["model"] == model.split("/")[-1])]
        errors = int((part["steps"] == 0).sum())
        print(f"{config:18s} {model:32s} задач {len(part):3d}  верно {part['correct'].mean():5.0%}  ошибок {errors:2d}  "
              f"цена ${part['cost'].sum():.3f}  шагов {part['steps'].mean():.1f}  потрачено сейчас ${ledger.total:.3f}", flush=True)
        if ledger.total > args.budget:
            print(f"бюджет ${args.budget} превышен, остановка", flush=True)
            break

    table = report(results[results["steps"] > 0])
    (out / "results.md").write_text(table.to_markdown(index=False), encoding="utf-8")
    money_chart(table, settings.img_dir / "homework.png")
    ledger.table().to_csv(out / "ledger.csv")
    (out / "run_meta.json").write_text(json.dumps({"tasks": len(tasks), "rows": len(results),
                                                   "errors": int((results["steps"] == 0).sum()),
                                                   "spent_this_run": round(ledger.total, 4),
                                                   "max_steps": args.max_steps, "workers": args.workers}, indent=1))
    print(table.to_string(index=False))
    print(f"потрачено за этот запуск ${ledger.total:.3f}")


if __name__ == "__main__":
    main()
