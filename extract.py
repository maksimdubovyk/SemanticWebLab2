"""П.2. LLM-екстракція структурованих фактів у facts.json через structured output Gemini.

Довгі статті діляться на сегменти за секціями: один виклик на всю статтю давав мало
фактів (v1, results/facts_v1.json). Модель наводить дослівну цитату (evidence), а код
перевіряє цитату й типи сутностей за схемою: некоректне йде у results/facts_rejected.json.
"""
import json
import re
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from llm import LLM_MODEL, generate_structured

ROOT = Path(__file__).parent
SEGMENT_CHARS = 8000
MAX_FACTS_PER_SEGMENT = 20


class EntityType(str, Enum):
    PERSON = "Person"
    ORGANIZATION = "Organization"      # компанія, університет, робоча група, проєкт
    TECHNOLOGY = "Technology"          # стандарт, мова, формат, протокол, програма
    KNOWLEDGE_BASE = "KnowledgeBase"   # база знань, онтологія, словник, набір даних


class Relation(str, Enum):
    CREATED = "CREATED"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    COLLABORATED_WITH = "COLLABORATED_WITH"
    STANDARDIZED_BY = "STANDARDIZED_BY"
    BUILT_ON = "BUILT_ON"
    DERIVED_FROM = "DERIVED_FROM"
    ACQUIRED = "ACQUIRED"
    SUPPORTS = "SUPPORTS"
    SUPERSEDED_BY = "SUPERSEDED_BY"
    INFLUENCED_BY = "INFLUENCED_BY"
    LINKED_TO = "LINKED_TO"


P, O, T, K = EntityType.PERSON, EntityType.ORGANIZATION, EntityType.TECHNOLOGY, EntityType.KNOWLEDGE_BASE
# Допустимі типи суб'єкта й об'єкта: єдине джерело правди для промпту і перевірки.
SIGNATURES: dict[Relation, tuple[set, set]] = {
    Relation.CREATED: ({P, O}, {T, K, O}),
    Relation.AFFILIATED_WITH: ({P}, {O}),
    Relation.COLLABORATED_WITH: ({P}, {P}),
    Relation.STANDARDIZED_BY: ({T, K}, {O}),
    Relation.BUILT_ON: ({T, K}, {T, K}),
    Relation.DERIVED_FROM: ({K, T}, {K, T}),
    Relation.ACQUIRED: ({O}, {O}),
    Relation.SUPPORTS: ({O}, {T, K}),
    Relation.SUPERSEDED_BY: ({T, K}, {T, K}),
    Relation.INFLUENCED_BY: ({T, K}, {T, K}),
    Relation.LINKED_TO: ({T, K}, {T, K}),
}


class Fact(BaseModel):
    subject: str = Field(description="Повна канонічна назва сутності, напр. 'Tim Berners-Lee'")
    subject_type: EntityType
    relation: Relation
    object: str = Field(description="Повна канонічна назва сутності")
    object_type: EntityType
    year: Optional[int] = Field(default=None, description="Рік події, лише якщо він є в тексті")
    evidence: str = Field(description="Дослівна цитата з тексту (одне речення), що підтверджує факт")


class ExtractionResult(BaseModel):
    facts: list[Fact]


PROMPT = """You extract a knowledge graph about the Semantic Web from part of a Wikipedia article.

Rules:
- Extract only facts stated in the text. Never use background knowledge.
- Both subject and object must be named entities of the allowed types; skip vague ones
  such as "researchers" or "the project". Named projects and working groups are Organization.
- Use full canonical names: "Tim Berners-Lee", not "Berners-Lee" or "he".
- Resolve what "it", "the term" or "the project" refers to from the surrounding text before
  writing a fact. In an article about X, "coined the term" means the term X.
- The object must be the thing actually acted on. Someone who designed the logo of X, filed a
  patent about X or wrote about X did not create X: skip such facts.
- `evidence` is one sentence copied verbatim from the text, with no omissions or "[...]".
- `year` is filled only when the text gives the year of this very fact.

Relations, always written in the direction subject -> object:
  CREATED: person or organization created, initiated or founded the object, or coined its name.
    Tim Berners-Lee CREATED World Wide Web Consortium
  AFFILIATED_WITH: person worked at, led or belonged to an organization.
    Kurt Bollacker AFFILIATED_WITH Metaweb
  COLLABORATED_WITH: two named people worked together or co-authored something.
    Douglas Lenat COLLABORATED_WITH Russell Greiner
  STANDARDIZED_BY: technology is standardized or published by an organization.
    SPARQL STANDARDIZED_BY World Wide Web Consortium
  BUILT_ON: subject uses, extends or is implemented on the object.
    Linked Data BUILT_ON HTTP
  DERIVED_FROM: subject takes its data from the object.
    YAGO DERIVED_FROM Wikidata
  ACQUIRED: subject organization bought the object organization.
    Google ACQUIRED Metaweb
  SUPPORTS: organization develops, funds, sponsors, hosts or adopts a technology or knowledge base.
    Cycorp SUPPORTS Cyc
  SUPERSEDED_BY: subject was replaced by the object. Both are projects or technologies, never companies.
    Freebase SUPERSEDED_BY Wikidata
  INFLUENCED_BY: subject was inspired by the object, without being built on it.
    Schema.org INFLUENCED_BY FOAF
  LINKED_TO: two knowledge bases are interlinked, neither depends on the other.
    YAGO LINKED_TO DBpedia

Return at most {max_facts} facts, preferring those that connect different entities.

Article: {title}

{text}
"""


def split_segments(text: str) -> list[str]:
    """Жадібно збирає секції верхнього рівня в сегменти до SEGMENT_CHARS символів."""
    sections = re.split(r"(?m)^(?===[^=])", text)
    segments = [""]
    for section in sections:
        if segments[-1] and len(segments[-1]) + len(section) > SEGMENT_CHARS:
            segments.append("")
        segments[-1] += section
    return segments


def check_types(fact: Fact) -> tuple[Optional[Fact], str]:
    """Перевіряє сигнатуру зв'язку. Обернений напрямок виправляється, решта відкидається."""
    subjects, objects = SIGNATURES[fact.relation]
    if fact.subject_type in subjects and fact.object_type in objects:
        return fact, "ok"
    if fact.object_type in subjects and fact.subject_type in objects:
        swapped = fact.model_copy(update={
            "subject": fact.object, "subject_type": fact.object_type,
            "object": fact.subject, "object_type": fact.subject_type,
        })
        return swapped, "swapped"
    return None, "type_mismatch"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def main() -> None:
    sources = json.loads((ROOT / "corpus" / "sources.json").read_text(encoding="utf-8"))
    facts: list[dict] = []
    rejected: list[dict] = []
    stats = {"swapped": 0, "type_mismatch": 0, "not_verbatim": 0, "duplicate": 0}
    for source in sources:
        text = (ROOT / "corpus" / f"{source['id']}.txt").read_text(encoding="utf-8")
        haystack = normalize_space(text)
        segments = split_segments(text)
        seen: set[tuple] = set()
        before = len(facts)
        for segment in segments:
            result = generate_structured(
                PROMPT.format(max_facts=MAX_FACTS_PER_SEGMENT, title=source["title"], text=segment),
                ExtractionResult,
            )
            for raw in result.facts:
                record = {"doc_id": source["id"], **raw.model_dump(mode="json")}
                fact, status = check_types(raw)
                if fact is None:
                    stats[status] += 1
                    rejected.append({**record, "reason": status})
                    continue
                if status == "swapped":
                    stats[status] += 1
                if normalize_space(fact.evidence) not in haystack:
                    stats["not_verbatim"] += 1
                    rejected.append({**record, "reason": "not_verbatim"})
                    continue
                key = (fact.subject.casefold(), fact.relation, fact.object.casefold())
                if key in seen:
                    stats["duplicate"] += 1
                    continue
                seen.add(key)
                facts.append({"id": f"F{len(facts) + 1:03d}", "doc_id": source["id"], **fact.model_dump(mode="json")})
        print(f"{source['id']:16} {len(segments):2} сегм. {len(facts) - before:3} фактів")

    (ROOT / "facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "facts_rejected.json").write_text(json.dumps(rejected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Разом {len(facts)} фактів ({LLM_MODEL}); перевірка: {stats}")


if __name__ == "__main__":
    main()
