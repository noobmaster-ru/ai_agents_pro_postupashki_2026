import pandas as pd

from rag.evaluation import diagnose, evaluate, final_answer, is_refusal, is_right, refusal_table, report

TASK = {"id": "t1", "source": "fresh", "question": "q", "answer": "15 March", "page": "P",
        "evidence": "15 March – Ukraine boycotts the closing ceremony of the Paralympics"}
GOLD_HIT = {"page": "P", "section": "March", "text": "15 March – Ukraine boycotts the closing ceremony of the Paralympics", "score": 0.8, "id": 1}
OTHER_HIT = {"page": "P", "section": "May", "text": "4 May – A car plows into a crowd in Leipzig", "score": 0.5, "id": 2}


def test_final_answer_takes_last_final_line_and_refusal():
    assert final_answer("thinking\nFINAL: 14 March\nFINAL: 15 March") == "15 March"
    assert is_refusal("NOT_FOUND") and is_refusal("Sources do not mention it.\nFINAL: NOT_FOUND")
    assert not is_refusal("The sources say NOT_FOUND is wrong.\nFINAL: 15 March")
    assert is_right(TASK, "[1] says so.\nFINAL: March 15, 2026") and not is_right({**TASK, "answer": ""}, "FINAL: anything")


def test_evaluate_marks_found_and_correct():
    def fn(question, model):
        return {"answer": "FINAL: 15 March", "hits": [OTHER_HIT, GOLD_HIT], "cost": 0.001, "steps": 1, "searches": 1}
    df = evaluate(fn, [TASK], "cfg", "openai/gpt-4o-mini", workers=1)
    row = df.iloc[0]
    assert row["correct"] and row["found"] and not row["refused"] and row["model"] == "gpt-4o-mini" and row["cost"] == 0.001


def test_evaluate_survives_exceptions_and_report_costs():
    def broken(question, model):
        raise RuntimeError("boom")
    df = evaluate(broken, [TASK], "cfg", "m", workers=1)
    assert not df.iloc[0]["correct"] and df.iloc[0]["answer"].startswith("ошибка")
    ok = pd.DataFrame([{"config": "c", "model": "m", "correct": True, "cost": 0.002, "searches": 1},
                       {"config": "c", "model": "m", "correct": False, "cost": 0.004, "searches": 1}])
    table = report(ok)
    assert table.iloc[0]["accuracy"] == 0.5 and table.iloc[0]["cost_per_task"] == 0.003 and table.iloc[0]["cost_per_correct"] == 0.006


def test_refusal_table_and_diagnose():
    answerable = pd.DataFrame([{"config": "c", "model": "m", "id": "a", "correct": False, "found": False, "refused": True},
                               {"config": "c", "model": "m", "id": "b", "correct": False, "found": True, "refused": False},
                               {"config": "c", "model": "m", "id": "d", "correct": True, "found": True, "refused": False}])
    none = pd.DataFrame([{"config": "c", "model": "m", "id": "n1", "refused": True}, {"config": "c", "model": "m", "id": "n2", "refused": False}])
    ref = refusal_table(answerable, none).iloc[0]
    assert ref["отклонено без ответа"] == "1 из 2" and ref["отклонено зря"] == "1 из 3"
    diag = diagnose(answerable, "c", "m")
    assert diag["поиск не нашёл"] == 1 and diag["нашёл, но ответ неверный"] == 1 and diag["ids не нашёл"] == ["a"]
