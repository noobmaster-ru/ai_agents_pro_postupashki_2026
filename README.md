# Курс по ИИ-агентам от поступашек, 2026

## Первая домашка

Лежит в [1_week_react/practice/](1_week_react/practice/), отчёт с таблицей, картинкой и разбором провалов в [1_week_react/practice/README.md](1_week_react/practice/README.md).

Что внутри:

- `data/fresh.jsonl` — 150 свежих вопросов о событиях 2026 года;
- `src/` — агент, инструменты, прогон и графики;
- `tests/` — тесты инструментов и проверки ответа;
- `results/` и `img/homework.png` — итоговая таблица и картинка;
- `traces/` — трейсы итогового прогона и вывод проверки свежести.

Запуск с чистого клона:

```
cd 1_week_react/practice
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                       # вписать OPENROUTER_API_KEY
python -m pytest -q tests                  # тесты, для Википедии нужен интернет
python check_fresh.py data/fresh.jsonl --sonnet
python run_homework.py --workers 4         # все конфигурации, около 2 часов и $4
python make_report.py results/results_raw.csv
```

Для пробного прогона `python run_homework.py --limit 10`.

## Вторая домашка

Лежит в [2_week_memory-embed-rag/](2_week_memory-embed-rag/), отчёт с графиком recall@k, таблицей гибрида, итоговой таблицей, таблицей отказов, разбором провалов и графиком токенов памяти в [2_week_memory-embed-rag/README.md](2_week_memory-embed-rag/README.md), выполненный ноутбук в [2_week_memory-embed-rag/homework02.ipynb](2_week_memory-embed-rag/homework02.ipynb).

Что внутри:

- `data/corpus.jsonl` — полные тексты 47 страниц Википедии про 2026 год, `data/questions.jsonl` — 148 вопросов с цитатами, `data/unanswerable.jsonl` — 10 вопросов без ответа, `data/embeddings_cache.npz` — кэш эмбеддингов;
- `rag/` — нарезка, кэш эмбеддингов, Milvus, поиск и гибрид, RAG-ответ, инструмент `knowledge_base`, память агента, замер и графики; агент и клиент OpenRouter берутся из первой домашки;
- `docker-compose.yml` — Milvus standalone;
- `tests/` — 20 тестов без модели и без ключа;
- `results/`, `img/` — таблицы и картинки итогового прогона.

Запуск с чистого клона:

```
cd 2_week_memory-embed-rag
python -m venv ../.venv && source ../.venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                       # вписать OPENROUTER_API_KEY
docker compose up -d                       # Milvus на localhost:19530
python -m pytest -q tests
python run_eval.py                         # прогоны, результаты кэшируются в results/
python build_notebook.py --run             # собрать и выполнить ноутбук
```

Без Docker: `MILVUS_URI=data/milvus_lite.db` переключает тот же код на Milvus Lite.

