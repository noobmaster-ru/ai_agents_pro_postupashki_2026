import pytest

from rag.hw1 import ToolRegistry
from rag.rag import KB_DESCRIPTION, KBArgs, knowledge_base_tool

HIT = {"id": 1, "page": "2026 in Japan", "section": "March", "text": "16 March – Two boats capsize off Henoko.", "score": 0.7}


def make_registry(search):
    registry = ToolRegistry()
    registry.register(knowledge_base_tool(search, k=5), KBArgs, KB_DESCRIPTION)
    return registry


def test_normal_input_returns_facts_with_page_and_section():
    registry = make_registry(lambda q, k, flt: [HIT])
    msg = registry.call({"id": "c1", "function": {"name": "knowledge_base", "arguments": '{"query": "boats Henoko"}'}})
    assert msg["role"] == "tool" and msg["content"] == "[2026 in Japan / March] 16 March – Two boats capsize off Henoko."


def test_page_filter_is_passed_and_empty_result_is_text():
    seen = {}

    def search(q, k, flt):
        seen.update(q=q, k=k, flt=flt)
        return []

    registry = make_registry(search)
    msg = registry.call({"id": "c2", "function": {"name": "knowledge_base", "arguments": '{"query": "x", "page": "2026 in Japan"}'}})
    assert msg["content"] == "nothing found" and seen == {"q": "x", "k": 5, "flt": 'page == "2026 in Japan"'}


def test_search_error_and_bad_arguments_become_text():
    def broken(q, k, flt):
        raise RuntimeError("milvus down")

    registry = make_registry(broken)
    msg = registry.call({"id": "c3", "function": {"name": "knowledge_base", "arguments": '{"query": "x"}'}})
    assert "не удалось вызвать инструмент knowledge_base" in msg["content"] and "milvus down" in msg["content"]
    bad = registry.call({"id": "c4", "function": {"name": "knowledge_base", "arguments": '{"nope": 1}'}})
    assert bad["content"].startswith("не удалось вызвать инструмент knowledge_base")


def test_tool_records_hits_per_call():
    tool = knowledge_base_tool(lambda q, k, flt: [HIT], k=5)
    tool.kb.reset()
    tool("a"); tool("b")
    assert len(tool.kb.hits) == 2
    with pytest.raises(TypeError):
        tool()
