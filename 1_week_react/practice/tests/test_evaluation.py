from src.evaluation import is_correct, final_answer


def fresh(answer):
    return {"source": "fresh", "answer": answer}


def test_final_answer():
    assert final_answer("думаю...\nFINAL: 15 March") == "15 March"
    assert final_answer("нет метки") == "нет метки"


def test_plain_containment():
    assert is_correct(fresh("Zohran Mamdani"), "FINAL: Zohran Mamdani")
    assert not is_correct(fresh("Zohran Mamdani"), "Brad Lander")


def test_date_order():
    assert is_correct(fresh("15 March"), "March 15, 2026")
    assert is_correct(fresh("15 March"), "15 March 2026")
    assert not is_correct(fresh("15 March"), "16 March")


def test_spelled_numbers():
    assert is_correct(fresh("Forty-one"), "41")
    assert is_correct(fresh("Twelve"), "12 people")
    assert is_correct(fresh("Thirty thousand people"), "30000 people")
    assert is_correct(fresh("nine"), "nine people")


def test_numbers_with_separators_and_units():
    assert is_correct(fresh("2,295"), "2295")
    assert is_correct(fresh("41.5 °C"), "41.5°C")
    assert is_correct(fresh("€16.4 billion"), "16.4 billion euros")
    assert is_correct(fresh("49.66%"), "49.66 percent")


def test_gsm8k_last_number():
    assert is_correct({"source": "gsm8k", "answer": "18"}, "Итого 18")
    assert not is_correct({"source": "gsm8k", "answer": "18"}, "18 яблок и 3 груши")


def test_word_boundaries():
    assert not is_correct(fresh("2"), "2026")
    assert not is_correct(fresh("22 March"), "122 March")
    assert is_correct(fresh("2"), "2")
    assert is_correct(fresh("two people"), "2 people")
    assert is_correct(fresh("four years"), "4 years")
