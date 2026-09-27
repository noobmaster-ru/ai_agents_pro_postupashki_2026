from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .hw1 import COLORS

PALETTE = [COLORS["violet"], COLORS["amber"], COLORS["teal"], COLORS["red"], COLORS["grey"]]


def _save(fig, path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def recall_chart(curves: dict[str, dict[int, float]], path=None, chosen_k: int | None = None):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for (name, curve), color in zip(curves.items(), PALETTE):
        ax.plot([str(k) for k in curve], [v * 100 for v in curve.values()], marker="o", lw=2.2, color=color, label=name)
    if chosen_k is not None:
        ks = list(next(iter(curves.values())))
        if chosen_k in ks:
            ax.axvline(ks.index(chosen_k), color=COLORS["grey"], ls="--", lw=1.2)
    ax.set_xlabel("k, сколько кусков берём"); ax.set_ylabel("recall@k, %"); ax.set_ylim(0, 105)
    ax.legend(frameon=False); ax.grid(alpha=0.3)
    return _save(fig, path)


def hybrid_chart(table: pd.DataFrame, path=None, metric: str = "recall@5"):
    pivot = table.pivot(index="нарезка", columns="поиск", values=metric)[["только вектор", "только слова", "гибрид RRF"]]
    ax = pivot.plot.bar(figsize=(7, 3.2), color=[COLORS["violet"], COLORS["teal"], COLORS["amber"]], rot=0)
    ax.set_ylabel(metric); ax.set_xlabel(""); ax.set_ylim(0, 1.05); ax.legend(frameon=False, loc="lower right"); ax.grid(alpha=0.3, axis="y")
    return _save(ax.figure, path)


def money_chart(table: pd.DataFrame, path=None, baseline: tuple[str, float] | None = None):
    fig, ax = plt.subplots(figsize=(9, 5))
    if baseline:
        ax.axhline(baseline[1] * 100, color=COLORS["grey"], ls="--", lw=1.2)
        ax.annotate(baseline[0], (0.01, baseline[1] * 100 + 1.5), xycoords=("axes fraction", "data"), fontsize=8, color=COLORS["grey"])
    offsets = [(8, 8), (8, -18), (-8, 8), (-8, -18)]
    for i, (_, r) in enumerate(table.iterrows()):
        color = COLORS["amber"] if "sonnet" in r["model"] else COLORS["teal"] if "haiku" in r["model"] else COLORS["violet"]
        ax.scatter(r["cost_per_task"] * 100, r["accuracy"] * 100, s=110, color=color)
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(f"{r['config']}\n{r['model']}", (r["cost_per_task"] * 100, r["accuracy"] * 100), fontsize=8,
                    xytext=(dx, dy), textcoords="offset points", ha="left" if dx > 0 else "right")
    ax.set_xscale("log"); ax.set_xlim(table["cost_per_task"].min() * 100 / 8, table["cost_per_task"].max() * 100 * 6)
    ax.set_ylim(-5, 105); ax.set_xlabel("цена вопроса, центы (логарифмическая шкала)"); ax.set_ylabel("доля верных, %"); ax.grid(alpha=0.3)
    return _save(fig, path)


def tokens_chart(per_turn: pd.DataFrame, path=None):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    labels = {"dialog: память": "окно из четырёх сообщений и факты", "dialog: вся история": "вся история в контексте"}
    for (tag, part), color in zip(per_turn.groupby("tag", sort=False), [COLORS["violet"], COLORS["amber"]]):
        ax.plot(range(1, len(part) + 1), part["prompt"].values, marker="o", lw=2.2, color=color, label=labels.get(tag, tag))
    ax.set_xlabel("реплика диалога"); ax.set_ylabel("токенов на входе"); ax.legend(frameon=False); ax.grid(alpha=0.3)
    return _save(fig, path)


def similarity_chart(answerable: list[float], unanswerable: list[float], path=None):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(answerable, bins=20, range=(0.2, 0.9), alpha=0.75, color=COLORS["teal"], label="ответ в корпусе есть")
    ax.hist(unanswerable, bins=20, range=(0.2, 0.9), alpha=0.75, color=COLORS["red"], label="ответа в корпусе нет")
    ax.set_xlabel("близость лучшего куска к вопросу"); ax.set_ylabel("вопросов"); ax.legend(frameon=False); ax.grid(alpha=0.3)
    return _save(fig, path)


def chunk_length_chart(line_chunks: list[dict], size: int = 400, path=None):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist([len(c["text"]) for c in line_chunks], bins=40, range=(0, 800), color=COLORS["violet"], alpha=0.85, label="по строкам")
    ax.axvline(size, color=COLORS["amber"], lw=2, label=f"наивная нарезка: ровно {size}")
    ax.set_xlabel("длина куска, символов"); ax.set_ylabel("кусков"); ax.legend(frameon=False); ax.grid(alpha=0.3)
    return _save(fig, path)
