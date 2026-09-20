import subprocess
import sys

from pydantic import BaseModel, Field

TIMEOUT = 5
OUTPUT_LIMIT = 5000


class ExecArgs(BaseModel):
    code: str = Field(description="код на Python; результат надо напечатать через print")


def python_exec(code: str) -> str:
    try:
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return f"Код превысил {TIMEOUT} сек"
    out = (r.stdout + r.stderr).strip()
    return out[:OUTPUT_LIMIT] if out else "Код исполнился, но результат пустой"
