import numpy as np

from rag.retrieval import build_keyword_index, is_gold, keyword_search, recall_curve, rrf, search_numpy


def test_rrf_prefers_items_high_in_both_lists():
    assert rrf([[1, 2, 3], [3, 1, 4]], k=2) == [1, 3]


def test_rrf_handles_single_and_empty_rankings():
    assert rrf([[5, 6]], k=5) == [5, 6]
    assert rrf([[], []], k=5) == []


def test_search_numpy_returns_sorted_top_k():
    vectors = np.eye(4, dtype=np.float32)
    top = search_numpy(np.array([0.1, 0.9, 0.3, 0.0], dtype=np.float32), vectors, k=2)
    assert [i for i, _ in top] == [1, 2]


def test_keyword_search_ranks_rare_terms():
    docs, idf = build_keyword_index(["earthquake in Japan", "election in Japan", "earthquake in Chile"])
    assert keyword_search("earthquake Chile", docs, idf, k=1) == [2]


def test_is_gold_and_recall_curve():
    task = {"page": "P", "answer": "Twelve", "evidence": "3 January – twelve people injured at a market"}
    chunks = [{"page": "P", "text": "3 January – 12 people injured at a market"}, {"page": "P", "text": "unrelated line about sport"},
              {"page": "Q", "text": "3 January – 12 people injured at a market"}]
    assert is_gold(chunks[0], task) and not is_gold(chunks[1], task) and not is_gold(chunks[2], task)
    assert recall_curve([[1, 0, 2]], chunks, [task], ks=(1, 3)) == {1: 0.0, 3: 1.0}
