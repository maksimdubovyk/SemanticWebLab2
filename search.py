"""П.3. Чанкінг, векторний індекс на numpy та порівняння keyword- і семантичного пошуку.

Розбиття: спершу за секціями статті, далі рекурсивно за абзацами й реченнями до
1000 символів (близько 170 слів, типовий абзац Вікіпедії), перекриття 150 символів.
Кожен чанк отримує префікс «стаття — секція», щоб фрагменти із займенниками не
втрачали контекст.
"""
import json
import re
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).parent
INDEX_DIR = ROOT / "index"
RESULTS = ROOT / "results"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 60
TOP_N = 5

# relevant_doc + relevant_text визначають, який чанк вважається правильним.
QUERIES = [
    {"expected_winner": "semantic", "query": "search giant bought the startup behind a community-built fact database",
     "relevant_doc": "freebase", "relevant_text": "Metaweb"},
    {"expected_winner": "semantic", "query": "machines that grasp what ordinary people take for granted",
     "relevant_doc": "cyc", "relevant_text": "common sense"},
    {"expected_winner": "semantic", "query": "body that sets the technical rules everyone follows when building websites",
     "relevant_doc": "w3c", "relevant_text": "standards"},
    {"expected_winner": "keyword", "query": "Q42775",
     "relevant_doc": "wikidata", "relevant_text": "Q42775"},
    {"expected_winner": "keyword", "query": "Georgi Kobilarov",
     "relevant_doc": "dbpedia", "relevant_text": "Kobilarov"},
    {"expected_winner": "keyword", "query": "Q1731",
     "relevant_doc": "semantic_web", "relevant_text": "Q1731"},
    {"expected_winner": "keyword", "query": "2,093,000",
     "relevant_doc": "cyc", "relevant_text": "2,093,000"},
]


def build_chunks() -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "], keep_separator="end",
    )
    chunks = []
    for source in json.loads((ROOT / "corpus" / "sources.json").read_text(encoding="utf-8")):
        text = (ROOT / "corpus" / f"{source['id']}.txt").read_text(encoding="utf-8")
        parts = re.split(r"^=+\s*(.+?)\s*=+\s*$", text, flags=re.MULTILINE)
        sections = [("Introduction", parts[0])] + list(zip(parts[1::2], parts[2::2]))
        for section, body in sections:
            for piece in splitter.split_text(body.strip()):
                if len(piece) < MIN_CHUNK_CHARS:
                    continue  # секція з одним рядком не несе змісту
                chunks.append({
                    "id": f"{source['id']}#{len(chunks)}",
                    "doc_id": source["id"],
                    "text": f"{source['title']} — {section}\n{piece.strip()}",
                })
    return chunks


class VectorIndex:
    """Власне сховище: numpy-матриця нормованих векторів і cosine similarity через dot."""

    def __init__(self, chunks: list[dict], vectors: np.ndarray):
        self.chunks = chunks
        self.vectors = vectors

    @classmethod
    def build(cls, chunks: list[dict]) -> "VectorIndex":
        from llm import embed
        vectors = np.array(embed([c["text"] for c in chunks], "RETRIEVAL_DOCUMENT"), dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        INDEX_DIR.mkdir(exist_ok=True)
        np.save(INDEX_DIR / "embeddings.npy", vectors)
        (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
        return cls(chunks, vectors)

    @classmethod
    def load(cls) -> "VectorIndex":
        chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        return cls(chunks, np.load(INDEX_DIR / "embeddings.npy"))

    def search(self, query: str, n: int = TOP_N) -> list[tuple[dict, float]]:
        """Запит -> топ-N релевантних чанків із косинусною подібністю."""
        from llm import embed
        vector = np.array(embed([query], "RETRIEVAL_QUERY")[0], dtype=np.float32)
        scores = self.vectors @ (vector / np.linalg.norm(vector))
        order = np.argsort(-scores)[:n]
        return [(self.chunks[i], float(scores[i])) for i in order]


def substring_search(chunks: list[dict], query: str, n: int = TOP_N) -> list[tuple[dict, float]]:
    needle = query.casefold()
    hits = [(c, float(c["text"].casefold().count(needle))) for c in chunks]
    return sorted((h for h in hits if h[1] > 0), key=lambda h: -h[1])[:n]


class TfidfIndex:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(c["text"] for c in chunks)

    def search(self, query: str, n: int = TOP_N) -> list[tuple[dict, float]]:
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        order = np.argsort(-scores)[:n]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]


def first_relevant_rank(hits: list[tuple[dict, float]], spec: dict) -> int | None:
    for rank, (chunk, _) in enumerate(hits, start=1):
        if chunk["doc_id"] == spec["relevant_doc"] and spec["relevant_text"].casefold() in chunk["text"].casefold():
            return rank
    return None


def main() -> None:
    chunks = build_chunks()
    sizes = [len(c["text"]) for c in chunks]
    print(f"Чанків: {len(chunks)}, символів: мін {min(sizes)}, медіана {int(np.median(sizes))}, макс {max(sizes)}")
    vector_index = VectorIndex.build(chunks)
    tfidf = TfidfIndex(chunks)

    methods = {
        "substring": lambda q: substring_search(chunks, q),
        "tfidf": tfidf.search,
        "semantic": vector_index.search,
    }
    rows, lines = [], ["# Порівняння пошуку", "", f"Топ-{TOP_N}; ранг першого релевантного чанка, «—» якщо його немає.", ""]
    lines += ["| Запит | Очікуваний переможець | substring | TF-IDF | semantic |", "| --- | --- | --- | --- | --- |"]
    details = []
    for spec in QUERIES:
        ranks, detail = {}, [f"\n## {spec['query']}\n"]
        for name, search in methods.items():
            hits = search(spec["query"])
            ranks[name] = first_relevant_rank(hits, spec)
            detail.append(f"**{name}** (ранг {ranks[name] or '—'})\n")
            detail += [f"{i}. `{c['id']}` ({score:.3f}): {c['text'][:140].replace(chr(10), ' / ')}…"
                       for i, (c, score) in enumerate(hits[:3], start=1)] or ["нічого не знайдено"]
            detail.append("")
        rows.append({**spec, "ranks": ranks})
        cells = " | ".join(str(ranks[m] or "—") for m in methods)
        lines.append(f"| {spec['query']} | {spec['expected_winner']} | {cells} |")
        details += detail
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "search_comparison.md").write_text("\n".join(lines + details) + "\n", encoding="utf-8")
    (RESULTS / "search_comparison.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
