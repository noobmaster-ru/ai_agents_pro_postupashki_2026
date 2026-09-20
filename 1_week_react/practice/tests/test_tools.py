import pytest
import requests

from src.tools import calculator, python_exec, web_search, page_find, default_registry
from src.tools import wiki as wiki_module


class TestCalculator:
    def test_normal(self):
        assert calculator("17*23+5") == "396"
        assert calculator("2**10") == "1024"
        assert calculator("7/2") == "3.5"

    def test_empty(self):
        assert calculator("").startswith("ошибка вычисления")

    def test_error(self):
        assert "ошибка" in calculator("__import__('os').system('ls')")


class TestPythonExec:
    def test_normal(self):
        assert python_exec("print(sum(range(10)))") == "45"

    def test_empty(self):
        assert python_exec("x = 1") == "Код исполнился, но результат пустой"

    def test_error(self):
        assert "ZeroDivisionError" in python_exec("1/0")
        assert "превысил" in python_exec("import time; time.sleep(10)")


class TestWebSearch:
    def test_normal(self):
        assert web_search("Scott Derrickson").startswith("[Scott Derrickson]")

    def test_empty(self):
        assert web_search("qwxzv 1234567 nonsense") == "No data"

    def test_error(self, monkeypatch):
        def broken(*args, **kwargs):
            raise requests.ConnectionError("down")
        monkeypatch.setattr(wiki_module.requests, "get", broken)
        monkeypatch.setattr(wiki_module.time, "sleep", lambda s: None)
        with pytest.raises(RuntimeError):
            web_search("anything")


class TestPageFind:
    def test_normal(self):
        out = page_find("2026 in Japan", "Haneda")
        assert out.startswith("[2026 in Japan]") and "Haneda" in out

    def test_empty(self):
        assert page_find("2026 in Japan", "qwxzvblorptk") == "[2026 in Japan] ключевые слова не найдены"
        assert page_find("Qwxzv nonexistent page 9876543", "x") == "No data"

    def test_error(self, monkeypatch):
        def broken(*args, **kwargs):
            raise requests.ConnectionError("down")
        monkeypatch.setattr(wiki_module.requests, "get", broken)
        monkeypatch.setattr(wiki_module.time, "sleep", lambda s: None)
        with pytest.raises(RuntimeError):
            page_find("2026 in Japan", "Haneda")


class TestRegistry:
    def test_call_returns_tool_message(self):
        registry = default_registry()
        msg = registry.call({"id": "c1", "function": {"name": "calculator", "arguments": '{"expr": "2+2"}'}})
        assert msg == {"role": "tool", "tool_call_id": "c1", "content": "4"}

    def test_bad_arguments_become_text(self):
        registry = default_registry()
        msg = registry.call({"id": "c2", "function": {"name": "calculator", "arguments": '{"nope": 1}'}})
        assert msg["content"].startswith("не удалось вызвать инструмент calculator")

    def test_unknown_tool(self):
        registry = default_registry()
        msg = registry.call({"id": "c3", "function": {"name": "ghost", "arguments": "{}"}})
        assert "ghost" in msg["content"]


def test_page_find_ranks_lines_by_keyword_hits():
    out = page_find("2026 in chess", "GCT Super Chess Classic Romania May")
    first_line = out.split("\n")[0]
    assert "Vincent Keymer" in first_line
