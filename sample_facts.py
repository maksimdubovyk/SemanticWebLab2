"""П.2. Випадкова вибірка з 20 фактів для ручної перевірки: results/verification.csv.

Колонки verdict, error_type, comment заповнюються вручну за оригінальним текстом.
"""
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent
SEED = 42
SAMPLE_SIZE = 20


def main() -> None:
    facts = json.loads((ROOT / "facts.json").read_text(encoding="utf-8"))
    sample = sorted(random.Random(SEED).sample(facts, SAMPLE_SIZE), key=lambda f: f["id"])
    out = ROOT / "results" / "verification.csv"
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        raise SystemExit(f"{out} вже існує: не перезаписую можливі ручні позначки.")
    with out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "doc_id", "fact", "year", "evidence", "verdict", "error_type", "comment"])
        for fact in sample:
            triple = f"{fact['subject']} ({fact['subject_type']}) -{fact['relation']}-> {fact['object']} ({fact['object_type']})"
            writer.writerow([fact["id"], fact["doc_id"], triple, fact["year"], fact["evidence"], "", "", ""])
    print(f"Вибірка: {out}")


if __name__ == "__main__":
    main()
