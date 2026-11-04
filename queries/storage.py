"""JSON storage of query sets, so that every method is evaluated on exactly
the same queries. Layout: data/queries/<graph>/size<N>_<density>.json"""

from __future__ import annotations

import json
from pathlib import Path

from queries.random_walk import Query
from search.graph import Graph

QUERY_DIR = Path(__file__).resolve().parent.parent / "data" / "queries"


def query_set_path(graph_name: str, size: int, density: str) -> Path:
    return QUERY_DIR / graph_name / f"size{size}_{density}.json"


def save_query_set(path: Path, queries: list[Query], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "queries": [
            {
                "labels": q.graph.labels.tolist(),
                "edges": [list(e) for e in q.graph.edges()],
                "origin": q.origin,
            }
            for q in queries
        ],
    }
    path.write_text(json.dumps(payload))


def load_query_set(path: Path) -> tuple[list[Query], dict]:
    payload = json.loads(path.read_text())
    queries = [
        Query(Graph.from_edges(len(d["labels"]), d["edges"], d["labels"]), d["origin"])
        for d in payload["queries"]
    ]
    return queries, payload["meta"]
