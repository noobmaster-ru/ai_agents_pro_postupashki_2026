import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .hw1 import WEEK1, Settings

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA, IMG, RESULTS = ROOT / "data", ROOT / "img", ROOT / "results"
EMBED_MODEL, DIM = "openai/text-embedding-3-small", 512
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
LITE_URI = str(DATA / "milvus_lite.db")


def load_env() -> None:
    for path in (ROOT / ".env", WEEK1 / ".env", ROOT.parent / ".env"):
        if path.exists():
            load_dotenv(path, override=False)


def make_settings() -> Settings:
    load_env()
    for folder in (DATA, IMG, RESULTS):
        folder.mkdir(exist_ok=True)
    return Settings.from_env(data_dir=DATA)


def load_jsonl(name: str | Path) -> list[dict]:
    path = Path(name) if Path(name).is_absolute() else DATA / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
