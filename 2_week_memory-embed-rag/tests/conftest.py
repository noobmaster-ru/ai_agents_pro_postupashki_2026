import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.config import DIM  # noqa: E402


def fake_embed(texts: list[str]) -> np.ndarray:
    matrix = np.zeros((len(texts), DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in text.lower().split():
            matrix[row, hash(word) % DIM] += 1.0
        if not matrix[row].any():
            matrix[row, 0] = 1.0
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


@pytest.fixture
def embed():
    return fake_embed


@pytest.fixture
def lite_store(tmp_path):
    from rag.store import VectorStore
    return VectorStore(uri=str(tmp_path / "test.db"))
