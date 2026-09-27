import time

import numpy as np
from pymilvus import MilvusClient

from .config import MILVUS_URI

FIELDS = ["page", "section", "url", "text", "line_no"]


class VectorStore:
    def __init__(self, uri: str = MILVUS_URI, token: str = "", db: str | None = None):
        self.uri = uri
        self.client = MilvusClient(uri=uri, token=token)
        if db and not self.is_lite:
            if db not in self.client.list_databases():
                self.client.create_database(db)
            self.client.use_database(db)

    @property
    def is_lite(self) -> bool:
        return not self.uri.startswith("http")

    def version(self) -> str:
        try:
            return self.client.get_server_version()
        except Exception:
            return "milvus-lite"

    def has(self, name: str) -> bool:
        return self.client.has_collection(name)

    def drop(self, name: str) -> None:
        if self.client.has_collection(name):
            self.client.drop_collection(name)

    def count(self, name: str) -> int:
        return int(self.client.get_collection_stats(name)["row_count"])

    def index(self, name: str, chunks: list[dict], vectors: np.ndarray, fields: list[str] = FIELDS, batch: int = 1000) -> int:
        self.drop(name)
        self.client.create_collection(name, dimension=int(vectors.shape[1]), metric_type="COSINE")
        rows = [{"id": int(c.get("id", i)), "vector": v.tolist(), **{f: c.get(f, "") for f in fields}}
                for i, (c, v) in enumerate(zip(chunks, vectors))]
        for i in range(0, len(rows), batch):
            self.client.insert(name, rows[i:i + batch])
        try:
            self.client.flush(name)
        except Exception:
            pass
        self.wait_index(name)
        return self.count(name)

    def wait_index(self, name: str, seconds: int = 180) -> bool:
        if self.is_lite:
            return True
        for _ in range(seconds):
            info = self.describe_index(name)
            if info.get("state") == "Finished" and int(info.get("pending_index_rows", 0)) == 0:
                self.client.load_collection(name)
                return True
            time.sleep(1)
        return False

    def search(self, name: str, vector: np.ndarray, k: int = 5, flt: str = "", fields: list[str] = FIELDS) -> list[dict]:
        return self.search_many(name, [vector], k, flt, fields)[0]

    def search_many(self, name: str, vectors, k: int = 5, flt: str = "", fields: list[str] = FIELDS) -> list[list[dict]]:
        data = [np.asarray(v, dtype=np.float32).tolist() for v in vectors]
        result = self.client.search(name, data=data, limit=k, filter=flt, output_fields=fields)
        return [[{**h["entity"], "id": int(h["id"]), "score": round(float(h["distance"]), 4)} for h in hits] for hits in result]

    def describe_index(self, name: str) -> dict:
        return dict(self.client.describe_index(name, "vector"))
