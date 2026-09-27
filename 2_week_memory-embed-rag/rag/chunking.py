import re
from typing import Callable


def chunk_lines(page: dict, min_len: int = 40) -> list[dict]:
    chunks, section = [], ""
    for n, raw in enumerate(page["text"].splitlines()):
        line = " ".join(raw.split())
        if line.startswith("="):
            section = line.strip("= ")
        elif len(line) > min_len:
            chunks.append({"page": page["page"], "section": section, "url": page["url"], "text": line, "line_no": n})
    return chunks


def chunk_chars(page: dict, size: int = 400) -> list[dict]:
    text = " ".join(page["text"].split())
    return [{"page": page["page"], "section": "", "url": page["url"], "text": text[i:i + size], "line_no": i // size}
            for i in range(0, len(text), size)]


def chunk_corpus(corpus: list[dict], fn: Callable[[dict], list[dict]] = chunk_lines, **kwargs) -> list[dict]:
    chunks = [c for page in corpus for c in fn(page, **kwargs)]
    for i, c in enumerate(chunks):
        c["id"] = i
    return chunks


def emb_text(chunk: dict) -> str:
    return f"{chunk['page']}: {chunk['text']}"


def words(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def torn_example(tasks: list[dict], char_chunks: list[dict]) -> tuple[dict, list[str]] | None:
    for task in tasks:
        evidence = re.findall(r"\w+", task["evidence"].lower())
        head, tail = " ".join(evidence[:6]), " ".join(evidence[-6:])
        parts = [c["text"] for c in char_chunks if c["page"] == task["page"] and (head in words(c["text"]) or tail in words(c["text"]))]
        if len(parts) == 2:
            return task, parts
    return None
