import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
OUT = HERE / "homework02.ipynb"

CELLS = [
    ("md", """# Домашнее задание 2. База знаний и память для агента

Корпус: полные тексты 47 обзорных страниц Википедии про 2026 год, из которых собраны вопросы первой домашки. Вопросы те же, 148 штук с цитатой `evidence`, плюс 10 вопросов без ответа в корпусе. Код лежит в пакете `rag/`, агент и клиент OpenRouter берутся из первой домашки (`1_week_react/practice/src`). Дорогие прогоны сделаны скриптом `run_eval.py` и закэшированы в `results/*.csv`, ноутбук их только читает; эмбеддинги берутся из кэша `data/embeddings_cache.npz`, так что повторный запуск ноутбука почти ничего не стоит.

Milvus поднят через `docker-compose.yml` (`docker compose up -d`). Если сервера нет, задайте `MILVUS_URI=data/milvus_lite.db` — тот же код заработает на Milvus Lite."""),
    ("code", """import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from rag import (DATA, IMG, RESULTS, MODELS, COLORS, EMBED_MODEL, DIM, make_settings, load_jsonl, Ledger, LLMClient,
                 chunk_corpus, chunk_lines, chunk_chars, emb_text, torn_example, EmbeddingCache, VectorStore,
                 search_numpy, rank_numpy, recall_curve, recall_at, is_gold, build_keyword_index, keyword_search, rrf)
from rag.rag import Retriever, RagAnswerer, PlainAnswerer, AgentAnswerer, make_registry, format_sources, KBArgs, KB_DESCRIPTION
from rag.evaluation import evaluate, report, refusal_table, diagnose, last_week_rows, final_answer
from rag.memory import EpisodicJournal, SemanticMemory, MemoryAssistant, extract_facts
from rag.viz import recall_chart, hybrid_chart, money_chart, tokens_chart, similarity_chart, chunk_length_chart
from rag.hw1 import WEEK1, show_trace

settings = make_settings()
ledger = Ledger()
client = LLMClient(settings, ledger)
corpus, tasks, unanswerable = load_jsonl("corpus.jsonl"), load_jsonl("questions.jsonl"), load_jsonl("unanswerable.jsonl")
len(corpus), sum(len(p["text"]) for p in corpus), len(tasks), len(unanswerable), tasks[0]"""),
    ("md", """## Шаг 1. Корпус и две нарезки

Наивная нарезка: по 400 символов подряд, не глядя на границы. Осмысленная: одна строка страницы это одно датированное событие, поэтому граница куска проходит по переводу строки, а заголовок раздела (месяц или тема) едет с куском как метаданные `section`. У каждого куска есть `page`, `section`, `url` и номер строки `line_no`, по которым человек найдёт источник."""),
    ("code", """line_chunks, char_chunks = chunk_corpus(corpus, chunk_lines), chunk_corpus(corpus, chunk_chars)
print(f"кусков по строкам: {len(line_chunks)}, по 400 символов: {len(char_chunks)}")
print(line_chunks[500])
task, parts = torn_example(tasks, char_chunks)
print("\\nфакт, который наивная нарезка разорвала:", task["question"], "| ответ:", task["answer"])
for n, part in enumerate(parts, 1):
    print(f"  кусок {n}: ...{part[-140:]}" if n == 1 else f"  кусок {n}: {part[:140]}...")
chunk_length_chart(line_chunks, path=IMG / "chunk_lengths.png");"""),
    ("md", """## Шаг 2. Эмбеддинги с кэшем

Ключ кэша это sha1 текста, значение вектор `text-embedding-3-small` на 512 измерений. Сначала загружается файл семинара `embeddings_512.npz` и наш `embeddings_cache.npz`, потом досчитывается только то, чего в них нет. Ниже видно, сколько текстов взято из кэша и сколько пришлось посчитать."""),
    ("code", """cache = EmbeddingCache(client, DATA / "embeddings_cache.npz", seeds=(DATA / "embeddings_512.npz",))
print("в кэше векторов:", cache.load())
line_vecs = cache.embed([emb_text(c) for c in line_chunks]); print("строки:", line_vecs.shape, "из кэша", cache.last_hits, "посчитано", cache.last_misses)
char_vecs = cache.embed([c["text"] for c in char_chunks]); print("символы:", char_vecs.shape, "из кэша", cache.last_hits, "посчитано", cache.last_misses)
q_vecs = cache.embed([t["question"] for t in tasks]); print("вопросы:", q_vecs.shape, "из кэша", cache.last_hits, "посчитано", cache.last_misses)
cache.save()
ledger.table()"""),
    ("md", """## Шаг 3. Качество поиска: recall@k по двум нарезкам

Верным считаем кусок с той же страницы, в котором есть ответ и не меньше половины слов исходной цитаты. Поиск здесь на numpy: векторы нормализованы, косинусная близость это одно умножение матрицы на вектор."""),
    ("code", """KS = (1, 3, 5, 10)
rank_lines, rank_chars = rank_numpy(q_vecs, line_vecs, 20), rank_numpy(q_vecs, char_vecs, 20)
curves = {"по строкам": recall_curve(rank_lines, line_chunks, tasks, KS), "по 400 символов": recall_curve(rank_chars, char_chunks, tasks, KS)}
K = 5
recall_chart(curves, path=IMG / "recall.png", chosen_k=K)
recall_df = pd.DataFrame(curves).round(3)
recall_df.to_csv(RESULTS / "recall.csv")
recall_df"""),
    ("md", """**Выбор k.** На нарезке по строкам кривая выходит на плато уже при k=5 (recall@5 и recall@10 совпадают), а каждый лишний кусок это ещё 60–80 токенов контекста на каждый вопрос, поэтому дальше везде k=5."""),
    ("md", """## Шаг 4. Индекс в Milvus и фильтр по метаданным

Обе нарезки кладём в отдельные коллекции. Рядом с вектором хранятся страница, раздел, ссылка, текст и номер строки. Проверяем, что выдача Milvus совпадает с перебором на numpy, и показываем фильтр по странице."""),
    ("code", """store = VectorStore()
print("Milvus:", store.uri, "| версия:", store.version())
for name, chunks, vecs in [("lines", line_chunks, line_vecs), ("chars", char_chunks, char_vecs)]:
    print(name, "векторов в коллекции:", store.index(name, chunks, vecs))
store.describe_index("lines")"""),
    ("code", """retriever = Retriever(store, cache, "lines", K)
pd.DataFrame(retriever.search(tasks[0]["question"], 3))[["score", "page", "section", "line_no", "text"]]"""),
    ("code", """from_numpy = [r[:5] for r in rank_lines]
from_milvus = [[h["id"] for h in hits] for hits in store.search_many("lines", q_vecs, 5)]
overlap = float(np.mean([len(set(a) & set(b)) / 5 for a, b in zip(from_numpy, from_milvus)]))
print(f"совпадение выдачи Milvus с перебором: {overlap:.1%} по top-5, полностью совпало {sum(a == b for a, b in zip(from_numpy, from_milvus))} из {len(tasks)} вопросов")"""),
    ("code", """flt = 'page == "2026 in Japan"'
print("фильтр:", flt)
pd.DataFrame(retriever.search("earthquake magnitude", 5, flt))[["score", "page", "section", "text"]]"""),
    ("code", """flt = 'page == "2026 in science" and section == "July"'
print("фильтр по странице и разделу:", flt)
pd.DataFrame(retriever.search("exoplanet atmosphere", 3, flt))[["score", "page", "section", "text"]]"""),
    ("md", """## Гибрид: слова + вектор, слияние рангов RRF

Вектор ловит смысл, поиск по словам с idf-весами ловит точные имена и числа. Reciprocal Rank Fusion складывает 1/(60+ранг) по обоим спискам. Считаем на обеих нарезках."""),
    ("code", """rows = []
for cut, chunks, ranked, texts in [("по 400 символов", char_chunks, rank_chars, [c["text"] for c in char_chunks]),
                                   ("по строкам", line_chunks, rank_lines, [emb_text(c) for c in line_chunks])]:
    docs, idf = build_keyword_index(texts)
    by_words = [keyword_search(t["question"], docs, idf, 20) for t in tasks]
    by_hybrid = [rrf([v, w], 20) for v, w in zip(ranked, by_words)]
    for name, r in [("только вектор", ranked), ("только слова", by_words), ("гибрид RRF", by_hybrid)]:
        rows.append({"нарезка": cut, "поиск": name, **{f"recall@{k}": round(recall_at(r, chunks, tasks, k), 3) for k in KS}})
hybrid = pd.DataFrame(rows)
hybrid.to_csv(RESULTS / "hybrid.csv", index=False)
hybrid_chart(hybrid, path=IMG / "hybrid.png", metric="recall@5")
hybrid"""),
    ("md", """## Шаг 5. RAG-ответ, агент с knowledge_base и замер

`RagAnswerer` кладёт k найденных кусков в контекст с номерами и просит ответить только по ним, `NOT_FOUND` если ответа нет. `AgentAnswerer` даёт агенту из первой домашки инструмент `knowledge_base(query, page)`: модель сама решает, что и сколько раз искать. Прогоны на всех 148 вопросах и 10 вопросах без ответа сделаны скриптом `run_eval.py` и закэшированы в `results/`; здесь один живой пример каждого и чтение результатов."""),
    ("code", """rag = RagAnswerer(client, retriever, k=K)
out = rag(tasks[0]["question"], MODELS["cheap"])
print(out["answer"], "| эталон:", tasks[0]["answer"], "| цена, центов:", round(out["cost"] * 100, 4))"""),
    ("code", """registry = make_registry(retriever, K)
print(json.dumps(registry["knowledge_base"].schema, ensure_ascii=False, indent=1))
agent_fn = AgentAnswerer(client, registry)
out = agent_fn(tasks[3]["question"], MODELS["cheap"])
print(final_answer(out["answer"]), "| эталон:", tasks[3]["answer"], "| обращений к базе:", out["searches"], "| цена, центов:", round(out["cost"] * 100, 4))"""),
    ("code", """names = {"plain_sonnet": "сильная модель, без инструментов", "plain_cheap": "дешёвая модель, без инструментов",
         "rag_cheap": "дешёвая модель, RAG всегда, k=5", "agent_cheap": "дешёвая модель, агент с knowledge_base",
         "rag_mid": "средняя модель, RAG всегда, k=5"}
frames, none_frames = [], []
for name, label in names.items():
    df, dn = pd.read_csv(RESULTS / f"{name}.csv"), pd.read_csv(RESULTS / f"{name}_unanswerable.csv")
    df["config"], dn["config"] = label, label
    frames.append(df); none_frames.append(dn)
live = last_week_rows(tasks, WEEK1 / "results" / "results_all.csv", "поиск + страница v2", "gpt-4o-mini", "дешёвая модель, живой поиск (1-я домашка)")
results = pd.concat(frames[:1] + [live] + frames[1:], ignore_index=True)
none_results = pd.concat(none_frames, ignore_index=True)
table = report(results)
table.to_csv(RESULTS / "report.csv", index=False)
(RESULTS / "report.md").write_text(table.to_markdown(index=False), encoding="utf-8")
table"""),
    ("code", """last_week = float(live["correct"].mean())
money_chart(table, IMG / "money.png", baseline=(f"прошлая неделя, агент с живым поиском по Википедии: {last_week:.0%}", last_week));"""),
    ("code", """refusals = refusal_table(results[results["config"].isin(names.values())], none_results)
refusals.to_csv(RESULTS / "refusals.csv", index=False)
(RESULTS / "refusals.md").write_text(refusals.to_markdown(index=False), encoding="utf-8")
refusals"""),
    ("code", """top_answerable = [search_numpy(q, line_vecs, 1)[0][1] for q in q_vecs]
top_unanswerable = [search_numpy(q, line_vecs, 1)[0][1] for q in cache.embed([t["question"] for t in unanswerable])]
similarity_chart(top_answerable, top_unanswerable, path=IMG / "similarity.png")
print("медиана близости лучшего куска: с ответом %.3f, без ответа %.3f" % (np.median(top_answerable), np.median(top_unanswerable)))"""),
    ("md", """## Шаг 6. Разбор провалов лучшей дешёвой конфигурации

Две кучки: «поиск не нашёл» (верного куска нет среди найденных) и «нашёл, но ответ неверный». Для каждой кучки один пример с тем, что было в контексте."""),
    ("code", """best = max(names.values(), key=lambda c: table[(table["config"] == c) & (table["model"] == "gpt-4o-mini")]["accuracy"].max() if "дешёвая" in c else -1)
print("лучшая дешёвая конфигурация:", best)
diag = diagnose(results, best, "gpt-4o-mini")
{k: v for k, v in diag.items() if not k.startswith("ids")}"""),
    ("code", """traces_name = next(n for n, l in names.items() if l == best)
traces = {t["task"]["id"]: t for t in map(json.loads, (RESULTS / f"{traces_name}_traces.jsonl").read_text(encoding="utf-8").splitlines())}
def show_failure(tid):
    t = traces[tid]
    print("вопрос:", t["task"]["question"], "\\nэталон:", t["task"]["answer"], "| цитата:", t["task"]["evidence"][:160])
    print("ответ модели:", final_answer(t["answer"]), "| поисков:", t.get("searches"))
    for h in t["hits"][:5]:
        print(f"  [{h['page']} / {h['section']}] {h['score']} {h['text'][:150]}")
print("=== поиск не нашёл ==="); show_failure(diag["ids не нашёл"][0])
print("\\n=== нашёл, но ответ неверный ==="); show_failure(diag["ids нашёл"][0])"""),
    ("code", """failures = results[(results["config"] == best) & (results["model"] == "gpt-4o-mini") & ~results["correct"]][["id", "found", "refused", "answer", "gold"]]
failures.to_csv(RESULTS / "failures_best.csv", index=False)
failures.head(20)"""),
    ("md", """## Шаг 7. Память агента

Эпизодическая память: журнал всех реплик в `data/episodes.jsonl` с сессией и временем, в контекст он не идёт. Семантическая: факты о пользователе, которые дешёвая модель извлекает в конце сессии и возвращает полным обновлённым списком, поэтому новый факт заменяет старый. Факты лежат в отдельной коллекции Milvus `facts`, к вопросу достаются три ближайших. Поиск по корпусу делается дважды, по вопросу и по вопросу с фактами, и списки сливаются через RRF."""),
    ("code", """journal = EpisodicJournal(DATA / "episodes.jsonl")
memory = SemanticMemory(store, cache.embed, "facts", DATA / "facts.json")
journal.clear(); memory.clear()
assistant = MemoryAssistant(client, retriever, memory, journal, MODELS["cheap"])
first = []
print(assistant.talk("s1", first, "Привет! Я Лена, продуктовый аналитик из Новосибирска. Слежу за космосом и хоккеем, отвечай коротко."))
print(assistant.talk("s1", first, "Что было в космосе в июле 2026 года?"))
print("\\nфакты после первой сессии:", assistant.end_session(first))"""),
    ("code", """second = []
print(assistant.talk("s2", second, "Что интересного случилось в моих темах за лето?"))
print(assistant.talk("s2", second, "Хоккей я забросила, теперь слежу за футболом. И я переехала в Калининград."))
print(assistant.talk("s2", second, "Кто выиграл чемпионат мира по футболу 2026?"))
print("\\nфакты после второй сессии:", assistant.end_session(second))
print("что вспоминается на вопрос про город:", memory.recall("в каком городе живёт пользователь", 1))
pd.DataFrame(journal.read())[["session", "time", "role", "text"]].assign(text=lambda d: d["text"].str[:80])"""),
    ("md", """**График токенов.** Один и тот же диалог из десяти вопросов двумя способами: «вся история в контексте» и «окно из четырёх сообщений плюс факты». Считаем токены на входе по репликам и суммарную цену."""),
    ("code", """SCRIPT = [t["question"] for t in tasks[40:50]]
for mode in ("память", "вся история"):
    history = []
    for q in SCRIPT:
        assistant.talk("bench", history, q, mode)
per_turn = pd.DataFrame([c for c in ledger.calls if c["tag"].startswith("dialog: ")][-2 * len(SCRIPT):])
tokens_chart(per_turn, path=IMG / "memory_tokens.png")
memory_cost = per_turn.groupby("tag", sort=False).agg(реплик=("prompt", "size"), токенов_на_входе=("prompt", "sum"), цена=("cost", "sum")).round(5)
memory_cost.to_csv(RESULTS / "memory_tokens.csv")
memory_cost"""),
    ("md", """## Бонус: тот же конвейер на PDF

Лекции курса и текст этого задания режутся по страницам, имя файла и номер страницы едут метаданными, ответ ссылается на файл и страницу."""),
    ("code", """from rag.pdf import pdf_corpus, COURSE_SYSTEM
pdf = pdf_corpus(Path.cwd())
print("страниц в PDF-корпусе:", len(pdf), "| файлы:", sorted({c["page"] for c in pdf}))
store.index("course", pdf, cache.embed([emb_text(c) for c in pdf])); cache.save()
pdf_retriever = Retriever(store, cache, "course", 3)
course_rag = RagAnswerer(client, pdf_retriever, k=3, system=COURSE_SYSTEM, tag="pdf")
for q in ["Что такое RRF и зачем он нужен?", "Сколько баллов дают за память: две сессии, замену факта, график токенов и тесты без модели?",
          "Какой бюджет на замер в пятидесяти вопросах в пяти конфигурациях?"]:
    out = course_rag(q, MODELS["cheap"])
    print(q, "\\n ", " ".join(out["answer"].split())[:300], "\\n  источники:", [(h["page"], h["section"]) for h in out["hits"]], "\\n")"""),
    ("code", """ledger.table(), round(ledger.total, 4)"""),
]


def main():
    nb = new_notebook(metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                "language_info": {"name": "python"}})
    nb["cells"] = [new_markdown_cell(src) if kind == "md" else new_code_cell(src) for kind, src in CELLS]
    nbformat.write(nb, OUT)
    print("written", OUT, len(nb["cells"]), "cells")
    if "--run" in sys.argv:
        from nbclient import NotebookClient
        client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(HERE)}})
        client.execute()
        nbformat.write(nb, OUT)
        print("executed", OUT)


if __name__ == "__main__":
    main()
