"""П.4. Граф знань із facts.json, спільноти Louvain, GraphRAG (map-reduce) проти vector-RAG."""
import json
from collections import Counter
from pathlib import Path

import community as community_louvain
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network

from llm import generate
from search import VectorIndex

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
SEED = 42
MIN_COMMUNITY_SIZE = 3
MAX_COMMUNITIES = 10
VECTOR_TOP_N = 8
VECTOR_TOP_N_WIDE = 20  # перевірка: чи не спричинена слабкість vector-RAG вузьким контекстом

# Різні написання однієї сутності, виявлені в facts.json. Версії YAGO об'єднано в один вузол.
ALIASES = {
    "Hendler": "James Hendler",
    "RDF": "Resource Description Framework",
    "RDF(S)": "RDF Schema",
    "European Organization for Nuclear Research": "CERN",
    "Wordnet": "WordNet",
    "Geonames": "GeoNames",
    "YAGO3": "YAGO",
    "YAGO4": "YAGO",
}

# Ручні правки помилок моделі, знайдених під час перегляду всіх фактів за текстом.
# facts.json не змінюється, щоб оцінка точності стосувалася самого виходу моделі.
DROP_FACTS = {
    ("OpenCyc", "SUPERSEDED_BY", "Cyc"),  # OpenCyc закрито, Cyc його не замінював
}
SWAP_FACTS = {
    ("SPARQL", "SUPERSEDED_BY", "RDQL"),  # RDQL є попередником SPARQL
    ("Resource Description Framework", "BUILT_ON", "Semantic Web"),
    ("Web Ontology Language", "BUILT_ON", "Semantic Web"),
}
RELABEL_FACTS = {
    ("SPARUL", "SUPERSEDED_BY", "SPARQL"): "BUILT_ON",  # розширення, а не застарілий проєкт
    ("Corese", "DERIVED_FROM", "RDF(S)"): "BUILT_ON",
}


def canonical(name: str) -> str:
    name = " ".join(name.split()).removeprefix("The ").removeprefix("the ")
    return ALIASES.get(name, name)


def apply_corrections(facts: list[dict]) -> list[dict]:
    corrected = []
    for fact in facts:
        key = (fact["subject"], fact["relation"], fact["object"])
        if key in DROP_FACTS:
            continue
        fact = dict(fact)
        if key in SWAP_FACTS:
            fact["subject"], fact["object"] = fact["object"], fact["subject"]
            fact["subject_type"], fact["object_type"] = fact["object_type"], fact["subject_type"]
        fact["relation"] = RELABEL_FACTS.get(key, fact["relation"])
        corrected.append(fact)
    return corrected


def build_graph(facts: list[dict]) -> nx.DiGraph:
    """Вузли — сутності; ребро зберігає всі зв'язки між парою сутностей."""
    graph = nx.DiGraph()
    for fact in apply_corrections(facts):
        subject, target = canonical(fact["subject"]), canonical(fact["object"])
        if subject == target:  # петлі на кшталт «DBpedia створила DBpedia» не несуть змісту
            continue
        graph.add_node(subject, type=fact["subject_type"])
        graph.add_node(target, type=fact["object_type"])
        if not graph.has_edge(subject, target):
            graph.add_edge(subject, target, facts=[])
        record = {"relation": fact["relation"], "year": fact["year"], "doc_id": fact["doc_id"], "fact_id": fact["id"]}
        known = {(f["relation"], f["year"]) for f in graph[subject][target]["facts"]}
        if (record["relation"], record["year"]) not in known:
            graph[subject][target]["facts"].append(record)
    return graph


def to_weighted_undirected(graph: nx.DiGraph) -> nx.Graph:
    """Louvain працює з неорієнтованим графом; вага ребра — кількість зв'язків."""
    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes)
    for source, target, data in graph.edges(data=True):
        weight = len(data["facts"])
        if undirected.has_edge(source, target):
            undirected[source][target]["weight"] += weight
        else:
            undirected.add_edge(source, target, weight=weight)
    return undirected


def draw(graph: nx.DiGraph, partition: dict[str, int]) -> None:
    sizes = Counter(partition.values())
    palette = plt.get_cmap("tab20")
    degree = dict(graph.degree())
    plt.figure(figsize=(18, 13))
    layout = nx.spring_layout(graph, seed=SEED, k=0.9)
    nx.draw_networkx_edges(graph, layout, alpha=0.25, arrows=False)
    nx.draw_networkx_nodes(
        graph, layout,
        node_size=[60 + 45 * degree[n] for n in graph],
        node_color=[palette(partition[n] % 20) for n in graph],
    )
    nx.draw_networkx_labels(graph, layout, labels={n: n for n in graph if degree[n] >= 3}, font_size=8)
    plt.title(f"Граф знань: {graph.number_of_nodes()} вузлів, {graph.number_of_edges()} ребер, {len(sizes)} спільнот")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(RESULTS / "graph.png", dpi=160)
    plt.close()

    network = Network(height="850px", width="100%", directed=True, cdn_resources="remote")
    for node in graph:
        network.add_node(node, label=node, group=partition[node], value=degree[node] + 1,
                         title=f"{graph.nodes[node]['type']}, спільнота {partition[node]}")
    for source, target, data in graph.edges(data=True):
        network.add_edge(source, target, title=", ".join(sorted({f["relation"] for f in data["facts"]})))
    network.write_html(str(RESULTS / "graph.html"), notebook=False)


def community_triples(graph: nx.DiGraph, members: set[str]) -> list[str]:
    """Включає і зовнішні ребра: вони показують зв'язки спільноти з рештою графа."""
    triples = []
    for source, target, data in graph.edges(data=True):
        if source in members or target in members:
            for fact in data["facts"]:
                year = f" ({fact['year']})" if fact["year"] else ""
                triples.append(f"{source} {fact['relation']} {target}{year}")
    return sorted(triples)


SUMMARY_PROMPT = """Below is one community from a knowledge graph about the Semantic Web.
Write a summary of 4-6 sentences in Ukrainian: what unites this community, who or what is
central in it, and how it connects to outside entities. Use only the listed facts.

Entities: {entities}

Facts:
{facts}
"""

GRAPHRAG_PROMPT = """You answer a global question from summaries of knowledge graph communities.
Combine the summaries into one answer in Ukrainian, naming concrete entities and drawing
generalizations across communities. Do not use outside knowledge.

Question: {question}

Community summaries:
{summaries}
"""

# Інструкції навмисно ідентичні GRAPHRAG_PROMPT: відрізнятися має лише контекст.
VECTOR_PROMPT = """You answer a global question from text fragments retrieved from a corpus.
Combine the fragments into one answer in Ukrainian, naming concrete entities and drawing
generalizations across fragments. Do not use outside knowledge.

Question: {question}

Fragments:
{chunks}
"""


def summarize_communities(graph: nx.DiGraph, partition: dict[str, int]) -> list[dict]:
    groups: dict[int, set[str]] = {}
    for node, community_id in partition.items():
        groups.setdefault(community_id, set()).add(node)
    ranked = sorted(groups.items(), key=lambda item: -len(item[1]))
    eligible = [item for item in ranked if len(item[1]) >= MIN_COMMUNITY_SIZE][:MAX_COMMUNITIES]
    print(f"Спільнот: {len(groups)}, резюмуємо {len(eligible)} (розмір >= {MIN_COMMUNITY_SIZE}, максимум {MAX_COMMUNITIES})")
    summaries = []
    for community_id, members in eligible:
        triples = community_triples(graph, members)
        summary = generate(SUMMARY_PROMPT.format(entities=", ".join(sorted(members)), facts="\n".join(triples)))
        summaries.append({"community": community_id, "size": len(members), "entities": sorted(members),
                          "facts_used": len(triples), "summary": summary})
    return summaries


def graphrag_answer(question: str, summaries: list[dict]) -> str:
    joined = "\n\n".join(f"Community {s['community']}:\n{s['summary']}" for s in summaries)
    return generate(GRAPHRAG_PROMPT.format(question=question, summaries=joined))


def vector_rag_answer(question: str, index: VectorIndex, n: int) -> tuple[str, set[str]]:
    hits = index.search(question, n)
    joined = "\n\n".join(chunk["text"] for chunk, _ in hits)
    return generate(VECTOR_PROMPT.format(question=question, chunks=joined)), {chunk["doc_id"] for chunk, _ in hits}


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    facts = json.loads((ROOT / "facts.json").read_text(encoding="utf-8"))
    graph = build_graph(facts)
    undirected = to_weighted_undirected(graph)
    partition = community_louvain.best_partition(undirected, weight="weight", random_state=SEED)
    print(f"Граф: {graph.number_of_nodes()} вузлів, {graph.number_of_edges()} ребер, "
          f"компонент зв'язності: {nx.number_connected_components(undirected)}, "
          f"модульність: {community_louvain.modularity(partition, undirected, weight='weight'):.3f}")
    draw(graph, partition)

    summaries = summarize_communities(graph, partition)
    (RESULTS / "community_summaries.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    index = VectorIndex.load()
    questions = [q for q in json.loads((ROOT / "questions.json").read_text(encoding="utf-8")) if q["kind"] == "global"]
    lines = ["# GraphRAG проти vector-RAG", ""]
    for question in questions:
        graph_answer = graphrag_answer(question["question"], summaries)
        lines += [
            f"## {question['id']}. {question['question']}", "",
            f"### GraphRAG ({len(summaries)} резюме спільнот)", "", graph_answer, "",
        ]
        for n in (VECTOR_TOP_N, VECTOR_TOP_N_WIDE):
            vector_answer, docs = vector_rag_answer(question["question"], index, n)
            lines += [
                f"### Vector-RAG (топ-{n} чанків із {len(docs)} документів: {', '.join(sorted(docs))})", "",
                vector_answer, "",
            ]
    (RESULTS / "graphrag_answers.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Готово: {RESULTS / 'graphrag_answers.md'}")


if __name__ == "__main__":
    main()
