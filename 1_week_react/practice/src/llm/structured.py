import json
import re

from pydantic import BaseModel, ValidationError

from .client import LLMClient
from .schemas import Answer

SCHEMA_PROMPT = ("Отвечай строго одним JSON-объектом по этой схеме, без текста вокруг, "
                 "строки в одну линию, без лишних отступов, \\t, \\n:\n{schema}")
RETRY_PROMPT = "Ответ не прошёл проверку структуры: {error}. Верни ответ строго по JSON-схеме."


def json_from(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("в ответе нет JSON")
    return json.loads(m.group(0), strict=False)


def schema_prompt(schema: type[BaseModel]) -> str:
    return SCHEMA_PROMPT.format(schema=json.dumps(schema.model_json_schema(), ensure_ascii=False))


def ask_structured(client: LLMClient, question: str, model: str, schema: type[BaseModel] = Answer,
                   attempts: int = 3, tag: str = "struct", run_id: str | None = None) -> BaseModel:
    messages = [{"role": "system", "content": schema_prompt(schema)}, {"role": "user", "content": question}]
    for _ in range(attempts):
        msg = client.chat(messages, model, tag=tag, run_id=run_id)
        try:
            return schema.model_validate(json_from(msg["content"]))
        except (ValidationError, ValueError) as e:
            messages += [msg, {"role": "user", "content": RETRY_PROMPT.format(error=e)}]
    raise ValueError(f"модель не вернула валидный {schema.__name__} за {attempts} попытки")
