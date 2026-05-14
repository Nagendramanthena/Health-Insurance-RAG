"""
Knowledge Graph Retriever for Health Insurance RAG.

Queries the NetworkX Knowledge Graph to find structured context
related to entities identified in the user query.
"""

import sys
import os

# Ensure project root is on sys.path so `config` can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import difflib
from langchain_core.documents import Document
from rich.console import Console

from config import GRAPH_DATA_PATH

console = Console()

class GraphRetriever:
    def __init__(self, graph_path: str = GRAPH_DATA_PATH):
        self.graph_path = graph_path
        self.G = self._load_graph()
        # Pre-cache node labels for faster fuzzy matching
        self.node_labels = list(self.G.nodes())

    def _load_graph(self):
        """Load the GraphML file into NetworkX."""
        if not os.path.exists(self.graph_path):
            console.print(f"[bold red]Warning:[/] Graph file not found at {self.graph_path}. Returning empty graph.")
            return nx.MultiDiGraph()
        return nx.read_graphml(self.graph_path)

    def _extract_entities(self, query: str, threshold: float = 0.7):
        """
        Simple entity extraction using fuzzy matching against node labels.
        In a production system, this could be an LLM-based NER step.
        """
        query_words = query.lower().split()
        found_entities = []

        for node in self.node_labels:
            node_lower = str(node).lower()
            if node_lower in query.lower():
                found_entities.append(node)
                continue

            # Fuzzy match for typos
            matches = difflib.get_close_matches(node_lower, query_words, n=1, cutoff=threshold)
            if matches:
                found_entities.append(node)

        return list(set(found_entities))

    def _get_node_context(self, node, max_rel: int = 15):
        """Format a node and its immediate neighbors into a string, with limits."""
        data = self.G.nodes[node]
        node_type = data.get("type", "Unknown")

        context = f"### GRAPH ENTITY: {node} ({node_type})\n"
        props = [f"{k}: {v}" for k, v in data.items() if k != "type"]
        if props:
            context += f"Properties: {', '.join(props)}\n"

        # Outgoing edges
        out_edges = list(self.G.out_edges(node, data=True))
        if out_edges:
            context += "Relationships:\n"
            for _, target, attr in out_edges[:max_rel]:
                rel = attr.get("relation", "connected to")
                target_type = self.G.nodes[target].get("type", "")
                context += f"  - [{rel}] -> {target} ({target_type})"
                rel_props = [f"{k}={v}" for k, v in attr.items() if k != "relation"]
                if rel_props:
                    context += f" [{', '.join(rel_props)}]"
                context += "\n"

            if len(out_edges) > max_rel:
                context += f"  ... and {len(out_edges) - max_rel} more outgoing relationships.\n"

        # Incoming edges
        in_edges = list(self.G.in_edges(node, data=True))
        if in_edges:
            rel_counts = {}
            for source, _, attr in in_edges:
                rel = attr.get("relation", "connected to")
                rel_counts[rel] = rel_counts.get(rel, 0) + 1

            shown_count = 0
            for source, _, attr in in_edges:
                rel = attr.get("relation", "connected to")
                if rel_counts[rel] > 5 and shown_count >= max_rel:
                    continue
                source_type = self.G.nodes[source].get("type", "")
                context += f"  - {source} ({source_type}) -> [{rel}] -> [THIS ENTITY]\n"
                shown_count += 1
                if shown_count >= max_rel:
                    break

            for rel, count in rel_counts.items():
                if count > 5:
                    context += f"  - [Note: {count} total nodes have '{rel}' relationship to this entity]\n"

        return context

    def invoke(self, query: str, k: int = 3) -> list[Document]:
        """Perform graph retrieval and return LangChain Documents."""
        entities = self._extract_entities(query)
        if not entities:
            return []

        docs = []
        for entity in entities[:k]:
            content = self._get_node_context(entity)
            metadata = {
                "source_file": "knowledge_graph.graphml",
                "doc_type": "knowledge_graph",
                "entity_name": entity,
                "chunk_type": "graph_node",
                "display_header": f">>> GRAPH CONTEXT: {entity} <<<\n"
            }
            docs.append(Document(page_content=content, metadata=metadata))

        return docs

if __name__ == "__main__":
    retriever = GraphRetriever()
    test_query = "What is the copay for metformin on the Silver plan and which doctors in Chicago accept it?"
    results = retriever.invoke(test_query)
    for d in results:
        print(d.page_content)
        print("-" * 40)
