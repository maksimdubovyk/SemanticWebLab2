"""П.1. Формує корпус: зв'язний текст статей англомовної Вікіпедії про Семантичний веб.

Результат: corpus/<id>.txt + corpus/sources.json (URL, ревізія, дата, кількість слів).
Секції зі списками посилань відкидаються: це не зв'язний текст.
"""
import json
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent
OUT = ROOT / "corpus"
API = "https://en.wikipedia.org/w/api.php"
PAUSE = 3  # секунди між запитами, щоб не отримувати 429
HEADERS = {"User-Agent": "semantic-web-course-lab/1.0 (educational corpus builder)"}

ARTICLES = {
    "semantic_web": "Semantic Web",
    "linked_data": "Linked data",
    "rdf": "Resource Description Framework",
    "owl": "Web Ontology Language",
    "sparql": "SPARQL",
    "berners_lee": "Tim Berners-Lee",
    "w3c": "World Wide Web Consortium",
    "dbpedia": "DBpedia",
    "wikidata": "Wikidata",
    "freebase": "Freebase (database)",
    "schema_org": "Schema.org",
    "knowledge_graph": "Knowledge graph",
    "yago": "YAGO (database)",
    "cyc": "Cyc",
}

DROP_SECTIONS = {
    "see also", "references", "external links", "further reading", "notes",
    "bibliography", "sources", "footnotes", "citations", "awards",
    "publications", "works",
}


def strip_sections(text: str) -> str:
    """Прибирає секції з DROP_SECTIONS разом з їхніми підсекціями."""
    kept, skip_level = [], None
    for line in text.splitlines():
        match = re.match(r"^(=+)\s*(.+?)\s*\1$", line.strip())
        if match:
            level = len(match.group(1))
            if skip_level is not None and level > skip_level:
                continue
            skip_level = level if match.group(2).lower() in DROP_SECTIONS else None
        if skip_level is None:
            kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def fetch(title: str) -> dict:
    params = {
        "action": "query", "format": "json", "formatversion": 2, "redirects": 1,
        "titles": title, "prop": "extracts|revisions|info", "inprop": "url",
        "explaintext": 1, "exsectionformat": "wiki", "rvprop": "ids|timestamp",
    }
    for attempt in range(5):
        response = requests.get(API, params=params, headers=HEADERS, timeout=30)
        if response.status_code != 429:
            break
        time.sleep(int(response.headers.get("Retry-After", 30 * (attempt + 1))))
    response.raise_for_status()
    return response.json()["query"]["pages"][0]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    sources = []
    for doc_id, title in ARTICLES.items():
        page = fetch(title)
        text = strip_sections(page["extract"])
        (OUT / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        revision = page["revisions"][0]
        sources.append({
            "id": doc_id, "title": page["title"], "url": page["fullurl"],
            "revision_id": revision["revid"], "revision_time": revision["timestamp"],
            "words": len(text.split()), "license": "CC BY-SA 4.0",
        })
        print(f"{doc_id:16} {sources[-1]['words']:6} слів")
        time.sleep(PAUSE)
    (OUT / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Разом: {len(sources)} документів, {sum(s['words'] for s in sources)} слів")


if __name__ == "__main__":
    main()
