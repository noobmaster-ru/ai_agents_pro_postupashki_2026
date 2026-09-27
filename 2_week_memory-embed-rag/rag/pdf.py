from pathlib import Path

from pypdf import PdfReader

from .rag import FINAL_RULES

COURSE_SYSTEM = ("Отвечай по-русски, используя только пронумерованные источники. После ответа укажи источник в виде "
                 "[файл, страница]. Если ответа в источниках нет, ответь ровно NOT_FOUND. " + FINAL_RULES)


def pdf_chunks(path: Path, min_len: int = 80) -> list[dict]:
    reader = PdfReader(str(path))
    pages = [(n + 1, " ".join((p.extract_text() or "").split())) for n, p in enumerate(reader.pages)]
    return [{"page": Path(path).name, "section": f"стр. {n}", "url": f"{Path(path).name}#page={n}", "text": text, "line_no": n}
            for n, text in pages if len(text) >= min_len]


def pdf_corpus(folder: Path, pattern: str = "*.pdf") -> list[dict]:
    chunks = [c for path in sorted(Path(folder).glob(pattern)) for c in pdf_chunks(path)]
    for i, c in enumerate(chunks):
        c["id"] = i
    return chunks
