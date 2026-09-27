import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from .embeddings import EmbeddingCache
from .hw1 import Agent, LLMClient, ToolRegistry
from .store import VectorStore

FINAL_RULES = ("The last line must be FINAL: <short answer>. The short answer is a single item, not a list: either a number in digits, "
               "or a date as day and month (for example 15 March), or a name, a title or a short phrase exactly as written in the source. "
               "Do not add explanations after FINAL:.")
FINISH_PROMPT = ("Tools are no longer available. Answer from what you already have, in English, "
                 "with the last line FINAL: <short answer> or FINAL: NOT_FOUND.")
RAG_SYSTEM = ("Answer the question using only the numbered sources. Cite the source number like [2]. "
              "If the sources do not contain the answer, reply exactly NOT_FOUND. " + FINAL_RULES)
PLAIN_SYSTEM = "Answer the question from your own knowledge. If you do not know, reply exactly NOT_FOUND. " + FINAL_RULES
AGENT_SYSTEM = ("You answer questions about events of 2026. Never answer from memory: look facts up with the knowledge_base tool. "
                "Facts live in Wikipedia year pages such as '2026 in Japan', '2026 in sports', '2026 in science', '2026 in politics' or '2026'. "
                "If the first result is off topic, search again with different or more specific words or restrict the page, "
                "but never repeat the same call. Stop searching once the fact is found. "
                "When done, write the last line as FINAL: <short answer>, or NOT_FOUND if the knowledge base has no answer. "
                "In FINAL write a number in digits, a date as day and month (15 March), a name or a short phrase as written in the source.")
KB_DESCRIPTION = ("Searches the local knowledge base of 2026 events, built from Wikipedia year pages, and returns the five closest "
                  "facts with their page and section. Use a short English query with names, places, dates and key terms of the event.")


class KBArgs(BaseModel):
    query: str = Field(description="short search query in English: names, dates, key terms of the event")
    page: str = Field(default="", description="optional exact page title to search within, for example '2026 in Japan'; empty means all pages")


@dataclass
class Retriever:
    store: VectorStore
    cache: EmbeddingCache
    collection: str = "lines"
    k: int = 5

    def search(self, query: str, k: int | None = None, flt: str = "") -> list[dict]:
        vector = self.cache.embed([query])[0]
        return self.store.search(self.collection, vector, k or self.k, flt)


def format_sources(hits: list[dict]) -> str:
    return "\n".join(f"[{i + 1}] ({h['page']}" + (f", {h['section']}" if h.get("section") else "") + f") {h['text']}"
                     for i, h in enumerate(hits))


@dataclass
class RagAnswerer:
    client: LLMClient
    retriever: Retriever
    k: int = 5
    system: str = RAG_SYSTEM
    tag: str = "rag"
    flt: str = ""

    def __call__(self, question: str, model: str) -> dict:
        hits = self.retriever.search(question, self.k, self.flt)
        run_id = uuid.uuid4().hex
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": f"Sources:\n{format_sources(hits)}\n\nQuestion: {question}"}]
        msg = self.client.chat(messages, model, tag=self.tag, run_id=run_id)
        return {"answer": msg.get("content") or "", "hits": hits, "cost": self.client.ledger.cost_of(run_id),
                "steps": 1, "searches": 1, "messages": messages + [msg]}


@dataclass
class PlainAnswerer:
    client: LLMClient
    system: str = PLAIN_SYSTEM
    tag: str = "plain"

    def __call__(self, question: str, model: str) -> dict:
        run_id = uuid.uuid4().hex
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": question}]
        msg = self.client.chat(messages, model, tag=self.tag, run_id=run_id)
        return {"answer": msg.get("content") or "", "hits": [], "cost": self.client.ledger.cost_of(run_id),
                "steps": 1, "searches": 0, "messages": messages + [msg]}


class KnowledgeBase:
    def __init__(self, search: Callable[[str, int, str], list[dict]], k: int = 5):
        self.search = search
        self.k = k
        self.local = threading.local()

    def reset(self) -> None:
        self.local.hits = []

    @property
    def hits(self) -> list[dict]:
        return getattr(self.local, "hits", [])

    def __call__(self, query: str, page: str = "") -> str:
        hits = self.search(query, self.k, f'page == "{page}"' if page else "")
        self.local.hits = self.hits + hits
        if not hits:
            return "nothing found"
        return "\n".join(f"[{h['page']} / {h.get('section', '')}] {h['text']}" for h in hits)


def knowledge_base_tool(search: Callable[[str, int, str], list[dict]], k: int = 5) -> Callable[[str, str], str]:
    kb = KnowledgeBase(search, k)

    def knowledge_base(query: str, page: str = "") -> str:
        return kb(query, page)

    knowledge_base.kb = kb
    return knowledge_base


def make_registry(retriever: Retriever, k: int = 5) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(knowledge_base_tool(retriever.search, k), KBArgs, KB_DESCRIPTION)
    return registry


@dataclass
class AgentAnswerer:
    client: LLMClient
    registry: ToolRegistry
    max_steps: int = 6
    system: str = AGENT_SYSTEM
    tool_names: tuple = ("knowledge_base",)
    traces: list = field(default_factory=list)

    def __call__(self, question: str, model: str) -> dict:
        kb = self.registry["knowledge_base"].fn.kb
        kb.reset()
        agent = Agent(self.client, self.registry, model, list(self.tool_names), self.max_steps, self.system, tag="agent",
                      finish_prompt=FINISH_PROMPT)
        run = agent.run(question)
        return {"answer": run.answer, "hits": list(kb.hits), "cost": run.cost, "steps": run.steps,
                "searches": run.tool_calls, "messages": run.messages}
