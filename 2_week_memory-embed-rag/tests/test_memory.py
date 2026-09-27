from rag.memory import EpisodicJournal, SemanticMemory


def test_empty_memory_recalls_nothing(lite_store, embed, tmp_path):
    memory = SemanticMemory(lite_store, embed, facts_file=tmp_path / "facts.json")
    assert memory.facts() == [] and memory.recall("где живёт пользователь") == []


def test_saved_fact_is_found(lite_store, embed, tmp_path):
    memory = SemanticMemory(lite_store, embed, facts_file=tmp_path / "facts.json")
    memory.save(["Лена живёт в Новосибирске", "Лена следит за хоккеем"])
    assert memory.recall("в каком городе живёт Лена", k=1) == ["Лена живёт в Новосибирске"]
    assert set(memory.facts()) == {"Лена живёт в Новосибирске", "Лена следит за хоккеем"}


def test_new_fact_replaces_old_one(lite_store, embed, tmp_path):
    memory = SemanticMemory(lite_store, embed, facts_file=tmp_path / "facts.json")
    memory.save(["Лена живёт в Новосибирске"])
    memory.save(["Лена живёт в Калининграде"])
    found = memory.recall("Лена живёт в городе", k=5)
    assert found == ["Лена живёт в Калининграде"]
    assert memory.facts() == ["Лена живёт в Калининграде"]


def test_journal_keeps_sessions_apart(tmp_path):
    journal = EpisodicJournal(tmp_path / "episodes.jsonl")
    journal.add("s1", "user", "привет"); journal.add("s2", "user", "снова привет")
    assert [r["text"] for r in journal.read("s1")] == ["привет"] and len(journal.read()) == 2
    assert {"session", "time", "role", "text"} <= set(journal.read()[0])
