import os
from dataclasses import dataclass
from pathlib import Path

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = {"cheap": "openai/gpt-4o-mini", "mid": "anthropic/claude-haiku-4.5", "strong": "anthropic/claude-sonnet-4.6"}
COLORS = {"violet": "#5436A3", "amber": "#F09000", "teal": "#00838F", "red": "#C43C3C", "grey": "#787882"}
DATA_CANDIDATES = [Path("data"), Path("qa_data"), Path("qa-data"),
                   Path("/kaggle/input/datasets/artmakar04/qa-data"), Path("/kaggle/input/seminar01-data/data")]


def load_api_key() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    if not os.getenv("OPENROUTER_API_KEY"):
        try:
            from kaggle_secrets import UserSecretsClient
            os.environ["OPENROUTER_API_KEY"] = UserSecretsClient().get_secret("OPENROUTER_API_KEY")
        except Exception:
            pass
    key = os.getenv("OPENROUTER_API_KEY", "").strip().strip("'\"")
    if not key:
        raise RuntimeError("нет ключа: положите его в .env рядом с ноутбуком или в Kaggle Secrets")
    return key


def find_data_dir(marker: str = "compare_10.jsonl") -> Path:
    return next((p for p in DATA_CANDIDATES if (p / marker).exists()), Path("data"))


@dataclass(frozen=True)
class Settings:
    api_key: str
    chat_url: str = CHAT_URL
    referer: str = "https://postypashki.ru"
    app_title: str = "agents-course-seminar01"
    data_dir: Path = Path("data")
    traces_dir: Path = Path("traces")
    img_dir: Path = Path("img")

    @classmethod
    def from_env(cls, data_dir: Path | None = None) -> "Settings":
        settings = cls(api_key=load_api_key(), data_dir=Path(data_dir) if data_dir else find_data_dir())
        settings.traces_dir.mkdir(exist_ok=True)
        settings.img_dir.mkdir(exist_ok=True)
        return settings

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "HTTP-Referer": self.referer, "X-Title": self.app_title}
