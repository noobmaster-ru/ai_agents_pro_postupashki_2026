import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
WEEK1 = HERE.parent / "1_week_react" / "practice"
QUESTIONS = WEEK1 / "data" / "fresh.jsonl"
WEEK1_RESULTS = WEEK1 / "results" / "results_all.csv"
LIVE_SEARCH_CONFIG = ("поиск + страница v2", "gpt-4o-mini")
WIKI = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "agents-course-homework02/1.0 (https://postypashki.ru; educational project)"}


def page_text(title: str) -> tuple[str, str]:
    for attempt in range(4):
        try:
            r = requests.get(WIKI, timeout=40, headers=HEADERS,
                             params={"action": "query", "prop": "extracts", "explaintext": 1, "titles": title,
                                     "redirects": 1, "format": "json"})
            if r.status_code == 200:
                page = next(iter(r.json()["query"]["pages"].values()))
                return page.get("title", title), page.get("extract", "")
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return title, ""


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", str(text).lower()).split())


def last_week_ok() -> dict[str, bool]:
    if not WEEK1_RESULTS.exists():
        return {}
    df = pd.read_csv(WEEK1_RESULTS)
    df = df[(df["config"] == LIVE_SEARCH_CONFIG[0]) & (df["model"] == LIVE_SEARCH_CONFIG[1])]
    return dict(zip(df["id"], df["correct"].astype(bool)))


def main() -> None:
    tasks = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    corpus = []
    for title in sorted({t["page"] for t in tasks}):
        real, text = page_text(title)
        if text:
            corpus.append({"page": real, "url": "https://en.wikipedia.org/wiki/" + real.replace(" ", "_"), "text": text})
        print(f"{real:42s} {len(text):8d} символов", flush=True)
    with (DATA / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for c in corpus:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    texts = {c["page"]: norm(c["text"]) for c in corpus}
    agent_ok = last_week_ok()
    kept, lost = [], []
    for t in tasks:
        (kept if norm(t["evidence"])[:120] in texts.get(t["page"], "") else lost).append(t)
    with (DATA / "questions.jsonl").open("w", encoding="utf-8") as f:
        for t in kept:
            row = {k: t[k] for k in ["id", "source", "question", "answer", "evidence", "page", "url"]}
            row["agent_ok"] = agent_ok.get(t["id"])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    lines = sum(len([l for l in c["text"].splitlines() if len(l.strip()) >= 40]) for c in corpus)
    print(f"\nстраниц: {len(corpus)}, символов: {sum(len(c['text']) for c in corpus)}, содержательных строк: {lines}")
    print(f"вопросов с цитатой, найденной в корпусе: {len(kept)} из {len(tasks)}")
    for t in lost:
        print(f"  потерян {t['id']}: {t['evidence'][:90]}")


if __name__ == "__main__":
    main()
