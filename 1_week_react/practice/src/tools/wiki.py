import time

import requests
from pydantic import BaseModel, Field

WIKI_URL = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "agents-course-seminar01/1.0 (https://postypashki.ru; educational project)"}
TEXT_LIMIT = 5000
NO_DATA = "No data"


class SearchArgs(BaseModel):
    query: str = Field(description="короткий поисковый запрос: имя, название, термин")


class PageFindArgs(BaseModel):
    title: str = Field(description="название статьи Википедии, например '2026 in Japan' или '2026 in sports'")
    keywords: str = Field(description="ключевые слова через пробел: месяц, имя, место, число; "
                                      "вернутся строки статьи, где есть хотя бы одно из них")


def wiki(params: dict, attempts: int = 3) -> dict:
    status = "без ответа"
    for attempt in range(attempts):
        try:
            r = requests.get(WIKI_URL, params={**params, "format": "json"}, headers=WIKI_HEADERS, timeout=20)
        except requests.RequestException as e:
            status = type(e).__name__
        else:
            status = r.status_code
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
                return r.json()["query"]
        time.sleep(1.0 + attempt)
    raise RuntimeError(f"Википедия ответила {status}")


def search_titles(query: str, limit: int = 1) -> list[str]:
    hits = wiki({"action": "query", "list": "search", "srsearch": query, "srlimit": limit})["search"]
    return [h["title"] for h in hits]


def page_extract(title: str, intro: bool) -> tuple[str, str]:
    params = {"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1, "titles": title}
    if intro:
        params["exintro"] = 1
    page = next(iter(wiki(params)["pages"].values()))
    return page.get("title", title), page.get("extract", "")


def web_search(query: str) -> str:
    titles = search_titles(query)
    if not titles:
        return NO_DATA
    title, extract = page_extract(titles[0], intro=True)
    return f"[{title}] {' '.join(extract.split())[:TEXT_LIMIT]}"


def page_find(title: str, keywords: str) -> str:
    found_title, extract = page_extract(title, intro=False)
    if not extract:
        titles = search_titles(title)
        if not titles:
            return NO_DATA
        found_title, extract = page_extract(titles[0], intro=False)
    if not extract:
        return NO_DATA
    words = {w.lower() for w in keywords.split() if w.strip()}
    lines = [" ".join(l.split()) for l in extract.splitlines() if l.strip()]
    scored = [(sum(w in l.lower() for w in words), i, l) for i, l in enumerate(lines)]
    found = [l for score, _, l in sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], s[1]))] if words else lines
    if not found:
        return f"[{found_title}] ключевые слова не найдены"
    return f"[{found_title}] " + "\n".join(found)[:TEXT_LIMIT]
