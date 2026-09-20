import time
import uuid
from dataclasses import dataclass, field

from ..llm.client import LLMClient
from ..tools.registry import ToolRegistry

SYSTEM = (
    "Ты решаешь задачи. Если нужно посчитать или найти факт, вызывай инструменты, а не угадывай. "
    "Вопросы могут быть о событиях 2025-2026 годов, которых нет в твоих знаниях: такие факты лежат в обзорных "
    "статьях англоязычной Википедии с названиями вида '2026 in Japan', '2026 in sports', '2026 in science', "
    "'2026 in politics' или просто '2026'. Сначала найди подходящую статью, затем читай её текст по ключевым "
    "словам из вопроса, если такой инструмент доступен. Не повторяй один и тот же вызов с теми же аргументами. "
    "Когда ответ готов, напиши его последней строкой в формате FINAL: <ответ>. "
    "В FINAL только сам ответ без пояснений: число цифрами (например 43, 2295, 41.5); если спрашивают "
    "сколько людей, лет, игр или месяцев, добавь это слово по-английски (2 people, 4 years, 8 games); "
    "дата как день и месяц по-английски (например 15 March); имя, название или короткая фраза так, "
    "как они написаны в источнике. Если факт найти не удалось, всё равно дай самый вероятный короткий ответ."
)
FINISH_PROMPT = ("Инструменты больше недоступны. Ответь по тому, что уже известно, "
                 "последней строкой FINAL: <ответ>.")


@dataclass
class Run:
    question: str
    answer: str
    steps: int
    messages: list = field(default_factory=list)
    cost: float = 0.0
    seconds: float = 0.0
    run_id: str = ""

    @property
    def tool_calls(self) -> int:
        return sum(m["role"] == "tool" for m in self.messages)


class Agent:
    def __init__(self, client: LLMClient, registry: ToolRegistry, model: str, tool_names: list[str] = (),
                 max_steps: int = 8, system: str = SYSTEM, tag: str = "agent"):
        self.client = client
        self.registry = registry
        self.model = model
        self.tool_names = list(tool_names)
        self.max_steps = max_steps
        self.system = system
        self.tag = tag

    def run(self, question: str) -> Run:
        run_id = uuid.uuid4().hex
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": question}]
        started = time.perf_counter()
        answer, steps = self._loop(messages, run_id)
        return Run(question, answer, steps, messages, self.client.ledger.cost_of(run_id),
                   time.perf_counter() - started, run_id)

    def _loop(self, messages: list, run_id: str) -> tuple[str, int]:
        tools = self.registry.schemas(self.tool_names) or None
        seen: set = set()
        for step in range(1, self.max_steps + 1):
            msg = self.client.chat(messages, self.model, tools=tools, tag=self.tag, run_id=run_id)
            messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                return msg.get("content") or "", step
            if self._looped(seen, calls) or step == self.max_steps:
                return self._finish(messages, step, run_id)
            messages += [self.registry.call(c) for c in calls]
        return "", self.max_steps

    def _finish(self, messages: list, step: int, run_id: str) -> tuple[str, int]:
        del messages[-1]
        messages.append({"role": "user", "content": FINISH_PROMPT})
        msg = self.client.chat(messages, self.model, tag=self.tag, run_id=run_id)
        messages.append(msg)
        return msg.get("content") or "", step + 1

    @staticmethod
    def _looped(seen: set, calls: list) -> bool:
        keys = [(c["function"]["name"], c["function"]["arguments"]) for c in calls]
        repeated = any(k in seen for k in keys)
        seen.update(keys)
        return repeated


def show_trace(run: Run) -> None:
    for m in run.messages[1:]:
        calls = "; ".join(f"{c['function']['name']}{c['function']['arguments']}" for c in m.get("tool_calls") or [])
        text = " ".join((m.get("content") or "").split())[:180]
        print(f"{m['role']:9s}| {text} {calls}")
    print(f"шагов: {run.steps}, цена: {run.cost * 100:.3f} ¢, время: {run.seconds:.1f} c")
