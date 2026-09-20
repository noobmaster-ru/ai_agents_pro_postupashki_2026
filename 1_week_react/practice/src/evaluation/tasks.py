import json
import re
from pathlib import Path

MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december")
UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
         "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}


def load_tasks(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(text) -> str:
    text = re.sub(r"[^\w\s]", " ", str(text).lower().replace(",", ""))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def words_to_number(words: list[str]) -> int | None:
    total, current = 0, 0
    for w in words:
        if w in UNITS:
            current += UNITS[w]
        elif w in TENS:
            current += TENS[w]
        elif w in SCALES:
            current = max(current, 1) * SCALES[w]
            if SCALES[w] >= 1000:
                total, current = total + current, 0
        else:
            return None
    return total + current


def spelled_numbers_to_digits(text: str) -> str:
    number_words = set(UNITS) | set(TENS) | set(SCALES)
    out, buffer = [], []
    for token in text.split() + [""]:
        if token in number_words:
            buffer.append(token)
            continue
        if buffer:
            value = words_to_number(buffer)
            out.append(str(value) if value is not None else " ".join(buffer))
            buffer = []
        if token:
            out.append(token)
    return " ".join(out)


def date_variants(text: str) -> set[str]:
    m = re.fullmatch(r"(\d{1,2}) (%s)" % "|".join(MONTHS), text) or re.fullmatch(r"(%s) (\d{1,2})" % "|".join(MONTHS), text)
    if not m:
        return set()
    day, month = (m.group(1), m.group(2)) if m.group(1).isdigit() else (m.group(2), m.group(1))
    return {f"{day} {month}", f"{month} {day}", f"{int(day)} {month}", f"{month} {int(day)}"}


def gold_variants(gold: str) -> set[str]:
    base = normalize(gold)
    variants = {base, spelled_numbers_to_digits(base)}
    variants |= date_variants(base)
    return {v for v in variants if v}


def is_correct(task: dict, answer) -> bool:
    if task["source"] == "gsm8k":
        nums = re.findall(r"-?\d+(?:\.\d+)?", str(answer).replace(",", ""))
        return bool(nums) and abs(float(nums[-1]) - float(task["answer"])) < 1e-6
    got = normalize(answer)
    return any(re.search(rf"(?<!\w){re.escape(v)}(?!\w)", got) for v in gold_variants(task["answer"]))


def final_answer(text: str) -> str:
    m = re.search(r"FINAL:\s*(.+)", text or "")
    return m.group(1).strip() if m else (text or "").strip()
