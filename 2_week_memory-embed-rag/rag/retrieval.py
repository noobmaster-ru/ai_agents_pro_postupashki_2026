import re

import numpy as np

NUMBERS = {word: str(n) for n, word in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}


def norm(text) -> str:
    tokens = re.sub(r"\b(a|an|the)\b|[^\w\s]", " ", str(text).lower()).split()
    return " ".join(NUMBERS.get(w, w) for w in tokens)


def is_gold(chunk: dict, task: dict) -> bool:
    if chunk["page"] != task["page"]:
        return False
    text, evidence = norm(chunk["text"]), norm(task["evidence"]).split()
    return norm(task["answer"]) in text and sum(w in text for w in evidence) >= 0.5 * len(evidence)


def search_numpy(query_vec: np.ndarray, vectors: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    sims = vectors @ query_vec
    top = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in top]


def rank_numpy(q_vecs: np.ndarray, vectors: np.ndarray, k: int = 20) -> list[list[int]]:
    sims = q_vecs @ vectors.T
    return [[int(i) for i in np.argsort(-row)[:k]] for row in sims]


def recall_at(rankings: list[list[int]], chunks: list[dict], tasks: list[dict], k: int) -> float:
    return float(np.mean([any(is_gold(chunks[i], t) for i in r[:k]) for r, t in zip(rankings, tasks)]))


def recall_curve(rankings: list[list[int]], chunks: list[dict], tasks: list[dict], ks=(1, 3, 5, 10)) -> dict[int, float]:
    return {k: recall_at(rankings, chunks, tasks, k) for k in ks}


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def build_keyword_index(texts: list[str]) -> tuple[list[set], dict[str, float]]:
    docs = [set(tokens(text)) for text in texts]
    df: dict[str, int] = {}
    for d in docs:
        for w in d:
            df[w] = df.get(w, 0) + 1
    return docs, {w: float(np.log(len(docs) / n)) for w, n in df.items()}


def keyword_search(query: str, docs: list[set], idf: dict[str, float], k: int = 20) -> list[int]:
    q = set(tokens(query))
    scores = np.array([sum(idf.get(w, 0.0) for w in q & d) for d in docs])
    return [int(i) for i in np.argsort(-scores)[:k]]


def rrf(rankings: list[list[int]], k: int = 5, K: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1 / (K + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)[:k]
