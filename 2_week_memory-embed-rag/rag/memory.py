import json
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np
from pydantic import BaseModel, Field

from .hw1 import LLMClient, json_from
from .rag import Retriever, format_sources
from .retrieval import rrf
from .store import VectorStore

FACTS_SYSTEM = ("Ты ведёшь память ассистента о пользователе. Верни один JSON-объект вида {\"facts\": [\"...\"]}: полный обновлённый список "
                "коротких фактов о пользователе. Добавь новые устойчивые факты: имя, город, работа, интересы, пожелания к ответам. "
                "Если новый факт отменяет старый, замени старый, а не добавляй рядом. Разовые вопросы и мелочи не записывай.")
ASSISTANT_SYSTEM = ("Ты помощник, который отвечает на вопросы о событиях 2026 года по базе знаний. Отвечай на языке пользователя, "
                    "опирайся на источники и на то, что знаешь о пользователе. Источники подобраны автоматически и могут быть не по теме: "
                    "используй только те, что относятся к вопросу. Если в источниках ответа нет, так и скажи.")


class Facts(BaseModel):
    facts: list[str] = Field(description="короткие факты о пользователе")


class EpisodicJournal:
    def __init__(self, path: Path):
        self.path = Path(path)

    def add(self, session: str, role: str, text: str) -> dict:
        record = {"session": session, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "role": role, "text": text}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def read(self, session: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        records = [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return [r for r in records if session is None or r["session"] == session]

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


class SemanticMemory:
    def __init__(self, store: VectorStore, embed: Callable[[list[str]], np.ndarray], collection: str = "facts",
                 facts_file: Path | None = None):
        self.store = store
        self.embed = embed
        self.collection = collection
        self.facts_file = Path(facts_file) if facts_file else None

    def facts(self) -> list[str]:
        if self.facts_file and self.facts_file.exists():
            return json.loads(self.facts_file.read_text(encoding="utf-8"))
        return []

    def save(self, facts: list[str]) -> list[str]:
        facts = list(dict.fromkeys(f.strip() for f in facts if f.strip()))
        if self.facts_file:
            self.facts_file.parent.mkdir(parents=True, exist_ok=True)
            self.facts_file.write_text(json.dumps(facts, ensure_ascii=False, indent=1), encoding="utf-8")
        self.store.drop(self.collection)
        if facts:
            rows = [{"id": i, "page": "memory", "section": "", "url": "", "text": f, "line_no": i} for i, f in enumerate(facts)]
            self.store.index(self.collection, rows, self.embed(facts))
        return facts

    def recall(self, question: str, k: int = 3) -> list[str]:
        if not self.store.has(self.collection) or self.store.count(self.collection) == 0:
            return []
        hits = self.store.search(self.collection, self.embed([question])[0], k)
        return [h["text"] for h in hits]

    def clear(self) -> None:
        self.save([])


def extract_facts(client: LLMClient, dialog: str, known: list[str], model: str, tag: str = "memory") -> list[str]:
    prompt = f"Известные факты: {json.dumps(known, ensure_ascii=False)}\n\nДиалог:\n{dialog}"
    msg = client.chat([{"role": "system", "content": FACTS_SYSTEM}, {"role": "user", "content": prompt}], model, tag=tag)
    return Facts.model_validate(json_from(msg.get("content") or "{}")).facts


class MemoryAssistant:
    def __init__(self, client: LLMClient, retriever: Retriever, memory: SemanticMemory, journal: EpisodicJournal,
                 model: str, window: int = 4, k: int = 5, system: str = ASSISTANT_SYSTEM):
        self.client, self.retriever, self.memory, self.journal = client, retriever, memory, journal
        self.model, self.window, self.k, self.system = model, window, k, system

    def search_with_memory(self, user_text: str, facts: list[str]) -> list[dict]:
        plain = self.retriever.search(user_text, self.k)
        personal = self.retriever.search(user_text + " " + " ".join(facts), self.k) if facts else []
        by_id = {h["id"]: h for h in plain + personal}
        return [by_id[i] for i in rrf([[h["id"] for h in plain], [h["id"] for h in personal]], self.k)]

    def talk(self, session: str, history: list, user_text: str, mode: str = "память") -> str:
        self.journal.add(session, "user", user_text)
        facts = self.memory.recall(user_text) if mode == "память" else []
        hits = self.search_with_memory(user_text, facts) if "?" in user_text else []
        sources = format_sources(hits) or "нет"
        system = self.system + ("\nЧто известно о пользователе: " + "; ".join(facts) if facts else "")
        window = history[-self.window:] if mode == "память" else history
        prompt = f"Источники:\n{sources}\n\n{user_text}"
        msg = self.client.chat([{"role": "system", "content": system}] + window + [{"role": "user", "content": prompt}],
                               self.model, tag=f"dialog: {mode}")
        answer = msg.get("content") or ""
        history += [{"role": "user", "content": user_text if mode == "память" else prompt}, {"role": "assistant", "content": answer}]
        self.journal.add(session, "assistant", answer)
        return answer

    def end_session(self, history: list) -> list[str]:
        dialog = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in history)
        return self.memory.save(extract_facts(self.client, dialog, self.memory.facts(), self.model))
