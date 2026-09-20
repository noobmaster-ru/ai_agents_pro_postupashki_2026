from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..agent.agent import Run
from ..config import COLORS
from ..llm.ledger import Ledger


def cost_time_chart(ledger: Ledger, tag: str):
    df = pd.DataFrame(ledger.by_tag(tag))
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.scatter(df["cost"] * 100, df["seconds"], s=140, color=COLORS["violet"])
    for _, r in df.iterrows():
        ax.annotate(r["model"], (r["cost"] * 100, r["seconds"]), xytext=(8, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("цена ответа, центы (логарифмическая шкала)")
    ax.set_ylabel("время ответа, секунды")
    ax.grid(alpha=0.3)
    return fig


def compare_chart(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    axes[0].bar(df["model"], df["accuracy"] * 100, color=COLORS["violet"])
    axes[0].set_ylabel("доля верных, %")
    axes[0].set_ylim(0, 105)
    axes[1].bar(df["model"], df["cost_per_correct"] * 100, color=COLORS["amber"])
    axes[1].set_ylabel("цена верного ответа, центы")
    axes[2].bar(df["model"], df["avg_seconds"], color=COLORS["teal"])
    axes[2].set_ylabel("секунд на задачу")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=12)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def context_chart(ledger: Ledger, run: Run, tag: str = "agent"):
    calls = ledger.by_tag(tag)[-run.steps:]
    steps = range(1, len(calls) + 1)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(steps, [c["prompt"] for c in calls], marker="o", color=COLORS["violet"], label="токены на входе")
    ax.plot(steps, [c["completion"] for c in calls], marker="o", color=COLORS["amber"], label="токены на выходе")
    ax.set_xlabel("шаг агента")
    ax.set_ylabel("токены за вызов")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def money_chart(table: pd.DataFrame, path: Path | str | None = None):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (_, r) in enumerate(table.iterrows()):
        color = COLORS["red"] if "каскад" in r["config"] else COLORS["amber"] if "sonnet" in r["model"] else COLORS["violet"]
        ax.scatter(r["cost_per_task"] * 100, r["accuracy"] * 100, s=110, color=color)
        left = i % 2 == 0
        ax.annotate(f"{r['config']}\n{r['model']}", (r["cost_per_task"] * 100, r["accuracy"] * 100), fontsize=8,
                    xytext=(-8 if left else 8, 6), textcoords="offset points", ha="right" if left else "left")
    ax.set_xscale("log")
    ax.set_xlim(table["cost_per_task"].min() * 100 / 8, table["cost_per_task"].max() * 100 * 6)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("цена задачи, центы (логарифмическая шкала)")
    ax.set_ylabel("доля верных ответов, %")
    ax.grid(alpha=0.3)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def steps_chart(results: pd.DataFrame):
    g = results.groupby("config").agg(steps=("steps", "mean"), tool_calls=("tool_calls", "mean"), seconds=("seconds", "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    g[["steps", "tool_calls"]].plot.bar(ax=axes[0], color=[COLORS["violet"], COLORS["teal"]], rot=12)
    axes[0].set_ylabel("в среднем на задачу")
    g["seconds"].plot.bar(ax=axes[1], color=COLORS["amber"], rot=12)
    axes[1].set_ylabel("секунд на задачу")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
        ax.set_xlabel("")
    fig.tight_layout()
    return fig


def repeats_chart(repeat_table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(repeat_table["config"], repeat_table["accuracy"] * 100, color=COLORS["violet"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("доля верных, %")
    ax.grid(alpha=0.3, axis="y")
    return fig


def cascade_chart(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(df["cost_per_task"] * 100, df["accuracy"] * 100, marker="o", color=COLORS["red"])
    for _, r in df.iterrows():
        ax.annotate(f"порог {r['threshold']}, наверх {r['escalated']:.0%}", (r["cost_per_task"] * 100, r["accuracy"] * 100),
                    xytext=(6, -12), textcoords="offset points", fontsize=8)
    ax.set_xlabel("цена задачи, центы")
    ax.set_ylabel("доля верных, %")
    ax.grid(alpha=0.3)
    return fig


def confidence_chart(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.hist(df[df["correct"]]["confidence"], bins=10, range=(0, 1), alpha=0.7, color=COLORS["teal"], label="верные")
    ax.hist(df[~df["correct"]]["confidence"], bins=10, range=(0, 1), alpha=0.7, color=COLORS["red"], label="неверные")
    ax.set_xlabel("заявленная уверенность")
    ax.set_ylabel("задач")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig
