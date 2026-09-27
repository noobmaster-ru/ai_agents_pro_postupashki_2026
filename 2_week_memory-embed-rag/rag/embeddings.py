import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .config import DIM, EMBED_MODEL
from .hw1 import LLMClient


class EmbeddingCache:
    def __init__(self, client: LLMClient | None, save_path: Path, seeds: tuple[Path, ...] = (), model: str = EMBED_MODEL,
                 dim: int = DIM, batch: int = 64, workers: int = 3):
        self.client = client
        self.save_path = Path(save_path)
        self.seeds = tuple(Path(p) for p in seeds)
        self.model, self.dim, self.batch, self.workers = model, dim, batch, workers
        self.vectors: dict[str, np.ndarray] = {}
        self.last_hits = self.last_misses = 0

    @staticmethod
    def key_of(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def load(self) -> int:
        for path in (*self.seeds, self.save_path):
            if path.exists():
                z = np.load(path)
                self.vectors.update({str(k): v.astype(np.float32) for k, v in zip(z["keys"], z["vectors"])})
        return len(self.vectors)

    def save(self) -> Path:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.save_path, keys=np.array(list(self.vectors)),
                            vectors=np.stack(list(self.vectors.values())).astype(np.float16))
        return self.save_path

    def __contains__(self, text: str) -> bool:
        return self.key_of(text) in self.vectors

    def __len__(self) -> int:
        return len(self.vectors)

    def embed(self, texts: list[str]) -> np.ndarray:
        missing = list(dict.fromkeys(t for t in texts if t not in self))
        self.last_hits, self.last_misses = len(texts) - len(missing), len(missing)
        if missing:
            self._fetch(missing)
        matrix = np.stack([self.vectors[self.key_of(t)] for t in texts])
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    def _fetch(self, missing: list[str]) -> None:
        if self.client is None:
            raise RuntimeError(f"{len(missing)} текстов нет в кэше эмбеддингов, а клиент не задан")
        batches = [missing[i:i + self.batch] for i in range(0, len(missing), self.batch)]

        def one(batch: list[str]) -> None:
            for text, vector in zip(batch, self.client.embed(batch, self.model, self.dim)):
                self.vectors[self.key_of(text)] = np.asarray(vector, dtype=np.float32)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(one, batches))
