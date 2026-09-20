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
