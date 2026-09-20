import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.evaluation import report
from src.viz import money_chart

ROWS = [
    ("без инструментов", "claude-sonnet-4.6", "сильная модель, без инструментов", "строка, которую надо обогнать"),
    ("без инструментов", "gpt-4o-mini", "дешёвая модель, без инструментов", "честный ноль: сколько дешёвая знает сама"),
    ("поиск", "gpt-4o-mini", "дешёвая модель, только поиск", "базовый агент с семинара"),
    ("поиск + страница", "gpt-4o-mini", "дешёвая модель, поиск и чтение страницы", "ваш лучший агент"),
    ("поиск + страница v2", "gpt-4o-mini", "дешёвая модель, поиск и чтение страницы, page_find v2", "тот же агент после починки page_find"),
    ("поиск + страница", "claude-haiku-4.5", "средняя модель, поиск и чтение страницы", "сколько добавляет модель посильнее"),
]
COLUMNS = ["config", "why", "model", "n", "accuracy", "cost_per_task", "cost_per_correct", "avg_steps", "avg_seconds"]


def homework_table(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config, model, label, why in ROWS:
        hit = table[(table["config"] == config) & (table["model"] == model)]
        if not hit.empty:
            rows.append({**hit.iloc[0].to_dict(), "config": label, "why": why})
    return pd.DataFrame(rows)[COLUMNS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="results_raw.csv из одного или нескольких прогонов")
    ap.add_argument("--out", default="results")
    ap.add_argument("--img", default="img/homework.png")
    args = ap.parse_args()

    frames = [pd.read_csv(p) for p in args.inputs]
    results = pd.concat(frames, ignore_index=True)
    results = results[results["steps"] > 0].drop_duplicates(subset=["config", "model", "id"], keep="last")
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    results.to_csv(out / "results_all.csv", index=False)

    table = homework_table(report(results))
    (out / "results.md").write_text(table.to_markdown(index=False), encoding="utf-8")
    Path(args.img).parent.mkdir(exist_ok=True)
    money_chart(table.drop(columns=["why"]), args.img)
    print(table.drop(columns=["why"]).to_string(index=False))
    print(f"строк: {len(results)}, суммарная цена ${results['cost'].sum():.3f}")


if __name__ == "__main__":
    main()
