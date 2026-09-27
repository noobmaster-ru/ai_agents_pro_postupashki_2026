from rag.chunking import chunk_chars, chunk_corpus, chunk_lines, emb_text

PAGE = {"page": "2026 in Testland", "url": "https://example.org/Testland", "text": "\n".join([
    "Events in the year 2026 in Testland, a small and quiet country.",
    "== January ==",
    "3 January – A magnitude 5.0 earthquake hits the capital, injuring twelve people at a market.",
    "short",
    "== February ==",
    "14 February – The parliament passes a new election law after a long night of debate in the chamber.",
])}


def test_lines_keep_section_and_skip_short():
    chunks = chunk_lines(PAGE)
    assert [c["section"] for c in chunks] == ["", "January", "February"]
    assert all(len(c["text"]) > 40 for c in chunks)
    assert chunks[1]["text"].startswith("3 January") and chunks[1]["url"] == PAGE["url"]


def test_chars_cover_whole_text():
    chunks = chunk_chars(PAGE, size=50)
    assert "".join(c["text"] for c in chunks) == " ".join(PAGE["text"].split())
    assert all(len(c["text"]) <= 50 for c in chunks)


def test_corpus_ids_and_embedding_text():
    chunks = chunk_corpus([PAGE, {**PAGE, "page": "2026 in Otherland"}], chunk_lines)
    assert [c["id"] for c in chunks] == list(range(len(chunks)))
    assert emb_text(chunks[0]) == "2026 in Testland: Events in the year 2026 in Testland, a small and quiet country."
