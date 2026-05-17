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

# ── Medical synonym expansion map ────────────────────────────
# Maps colloquial / lay terms → canonical specialty names in the graph
SYNONYM_MAP = {
    "brain doctor": "Neurology",    "brain doctors": "Neurology",
    "brain specialist": "Neurology", "neurologist": "Neurology", "neuro": "Neurology",
    "heart doctor": "Cardiology",   "heart doctors": "Cardiology",
    "heart specialist": "Cardiology", "cardiologist": "Cardiology",
    "eye doctor": "Ophthalmology",  "eye doctors": "Ophthalmology",
    "eye specialist": "Ophthalmology", "optometrist": "Ophthalmology",
    "skin doctor": "Dermatology",   "skin doctors": "Dermatology",
    "dermatologist": "Dermatology",
    "bone doctor": "Orthopedics",   "joint specialist": "Orthopedics",
    "orthopedist": "Orthopedics",
    "gut doctor": "Gastroenterology", "stomach doctor": "Gastroenterology",
    "gastro": "Gastroenterology",
    "kidney doctor": "Nephrology",  "kidney specialist": "Nephrology",
    "lung doctor": "Pulmonology",   "lung specialist": "Pulmonology",
    "kids doctor": "Pediatrics",    "child doctor": "Pediatrics",
    "pediatrician": "Pediatrics",
    "diabetes doctor": "Endocrinology", "endocrinologist": "Endocrinology",
    "women doctor": "OB/GYN",       "gynecologist": "OB/GYN", "obgyn": "OB/GYN",
    "blood doctor": "Hematology",
    "cancer doctor": "Oncology",    "oncologist": "Oncology",
    "psychiatrist": "Psychiatry",   "mental health": "Psychiatry",
}

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

    def _expand_synonyms(self, query: str) -> str:
        """Replace lay medical terms with canonical graph entity names."""
        q = query
        # Sort by length descending so multi-word phrases match before single words
        for phrase, canonical in sorted(SYNONYM_MAP.items(), key=lambda x: -len(x[0])):
            import re
            q = re.sub(rf'\b{re.escape(phrase)}\b', canonical, q, flags=re.IGNORECASE)
        return q

    def _extract_entities(self, query: str, threshold: float = 0.7):
        """
        Entity extraction: first expands medical synonyms, then matches
        against node labels in the knowledge graph.
        Handles both single-word tokens and multi-word location nodes like 'New York, NY'.
        """
        expanded_query = self._expand_synonyms(query)
        q_lower = expanded_query.lower()
        query_words = q_lower.split()
        found_entities = []

        import re
        for node in self.node_labels:
            node_str  = str(node)
            node_lower = node_str.lower()

            # For Location nodes like "New York, NY" — strip the ", ST" suffix
            # and check if the city portion appears as a phrase in the query
            node_data = self.G.nodes.get(node, {})
            if node_data.get("type") == "Location":
                city_part = node_lower.split(",")[0].strip()  # e.g. "new york"
                # Use word-boundary on the city part only
                if re.search(rf"\b{re.escape(city_part)}\b", q_lower):
                    found_entities.append(node)
                    continue
            else:
                # Standard whole-word match
                if re.search(rf"\b{re.escape(node_lower)}\b", q_lower):
                    found_entities.append(node)
                    continue

                # Fuzzy match for single-word tokens (typos)
                matches = difflib.get_close_matches(node_lower, query_words, n=1, cutoff=threshold)
                if matches:
                    found_entities.append(node)

        return list(set(found_entities))

    def _get_node_context(self, node, max_rel: int = 15):
        """Format a node and its immediate neighbors into a string, with limits."""
        if node not in self.G:
            return ""
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
                target_data = self.G.nodes[target]
                target_type = target_data.get("type", "")
                target_name = target_data.get("name", "")
                
                display_target = f"{target_name} ({target})" if target_name else str(target)
                context += f"  - [{rel}] -> {display_target} ({target_type})"
                
                # Include some properties of the target node if it's a Provider or PlanTier
                if target_type in ["Provider", "PlanTier", "Location"]:
                    t_props = [f"{k}={v}" for k, v in target_data.items() if k not in ["type", "name"]]
                    if t_props:
                        context += f" [{', '.join(t_props)}]"
                
                rel_props = [f"{k}={v}" for k, v in attr.items() if k != "relation"]
                if rel_props:
                    context += f" (Edge props: {', '.join(rel_props)})"
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
                
                source_data = self.G.nodes[source]
                source_type = source_data.get("type", "")
                source_name = source_data.get("name", "")
                
                display_source = f"{source_name} (NPI: {source})" if source_name and source_type == "Provider" else str(source)
                context += f"  - {display_source} ({source_type}) -> [{rel}] -> [THIS ENTITY]"
                
                # Include location if the source is a provider
                if source_type == "Provider":
                    loc = []
                    if "city" in source_data: loc.append(source_data["city"])
                    if "state" in source_data: loc.append(source_data["state"])
                    if loc:
                        context += f" [Location: {', '.join(loc)}]"
                
                context += "\n"
                shown_count += 1
                if shown_count >= max_rel:
                    break

            for rel, count in rel_counts.items():
                if count > 5:
                    context += f"  - [Note: {count} total nodes have '{rel}' relationship to this entity]\n"

        return context

    def _get_intersection_context(self, entities: list[str]):
        """Find common neighbors or paths between multiple entities."""
        if len(entities) < 2:
            return ""
        
        # Get all neighbors (predecessors and successors) for each entity
        neighbor_sets = []
        for e in entities:
            if e in self.G:
                ns = set(self.G.neighbors(e)) | set(self.G.predecessors(e))
                neighbor_sets.append(ns)
            else:
                neighbor_sets.append(set())
        
        common = set.intersection(*neighbor_sets) if neighbor_sets else set()
        
        if not common:
            return "" # Return nothing if no intersection; the individual node contexts will still be provided

        context = f"### GRAPH INTERSECTION: {', '.join(entities)}\n"
        context += f"The following entities are related to ALL of: {', '.join(entities)}:\n"
        for node in list(common)[:10]:
            context += self._get_node_context(node, max_rel=5)
            context += "---\n"
        return context

    def _location_provider_fallback(self, location_node: str, specialty_entities: list) -> str:
        """
        3-tier geographic cascade:
          1. Exact city match (already handled by node context)
          2. Same state — providers in the same state with matching specialty
          3. Any state — nearest available providers nationwide with that specialty
        """
        loc_data = self.G.nodes.get(location_node, {})
        state = loc_data.get("state", "")
        city  = loc_data.get("city", "")

        # Resolve target specialties
        target_specs = [
            n for n in specialty_entities
            if self.G.nodes.get(n, {}).get("type") == "Specialty"
        ]

        def _find_providers(state_filter=None):
            return [
                (n, d) for n, d in self.G.nodes(data=True)
                if d.get("type") == "Provider"
                and (not target_specs or d.get("specialty", "") in target_specs)
                and (state_filter is None or d.get("state", "") == state_filter)
            ]

        # ── Tier 1: same state ────────────────────────────────────────────────
        if state:
            same_state = _find_providers(state_filter=state)
            # Exclude providers already IN the requested city (they'd appear via node context)
            same_state = [(n,d) for n,d in same_state if d.get("city","") != city]
            if same_state:
                spec_label = target_specs[0] if target_specs else "Specialty"
                context = f"### NEARBY PROVIDERS IN {state} (no exact match in {city})\n"
                context += f"Note: No in-network {spec_label} providers found in {city}. "
                context += f"Showing the closest in-network options in {state}:\n\n"
                for npi, d in same_state[:8]:
                    context += (f"- {d.get('name','Unknown')} | {d.get('specialty','')} | "
                               f"{d.get('city','')}, {d.get('state','')} (NPI: {npi})\n")
                return context

        # ── Tier 2: nationwide (any state) ───────────────────────────────────
        nationwide = _find_providers()
        if nationwide:
            spec_label = target_specs[0] if target_specs else "requested specialty"
            # Group by state for a cleaner response
            by_state: dict = {}
            for npi, d in nationwide:
                s = d.get("state", "Unknown")
                by_state.setdefault(s, []).append((npi, d))

            context = f"### IN-NETWORK PROVIDERS — NATIONWIDE RESULTS\n"
            context += (f"Note: No in-network {spec_label} providers were found in {city or location_node} "
                       f"or nearby. Showing all available in-network options:\n\n")
            for st, providers in list(by_state.items())[:5]:
                context += f"**{st}:**\n"
                for npi, d in providers[:4]:
                    context += (f"  - {d.get('name','Unknown')} | {d.get('specialty','')} | "
                               f"{d.get('city','')}, {st} (NPI: {npi})\n")
            return context

        return ""

    def invoke(self, query: str, k: int = 3) -> list[Document]:
        """Perform graph retrieval and return LangChain Documents."""
        entities = self._extract_entities(query)
        if not entities:
            return []

        docs = []
        
        # 1. Add intersection context if multiple entities found
        if len(entities) >= 2:
            intersection_content = self._get_intersection_context(entities)
            if intersection_content:
                metadata = {
                    "source_file": "knowledge_graph.graphml",
                    "doc_type": "knowledge_graph",
                    "entity_name": "intersection",
                    "chunk_type": "graph_intersection",
                    "display_header": f">>> GRAPH INTERSECTION: {', '.join(entities)} <<<\n"
                }
                docs.append(Document(page_content=intersection_content, metadata=metadata))

        # 2. Add individual node contexts
        location_nodes = [e for e in entities if self.G.nodes.get(e, {}).get("type") == "Location"]
        specialty_nodes = [e for e in entities if self.G.nodes.get(e, {}).get("type") == "Specialty"]
        provider_nodes  = [e for e in entities if self.G.nodes.get(e, {}).get("type") == "Provider"]

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

        # 3. State-level fallback: if we have a location but no matching providers in that city
        if location_nodes and not provider_nodes:
            for loc in location_nodes:
                fallback_content = self._location_provider_fallback(loc, specialty_nodes)
                if fallback_content:
                    docs.append(Document(
                        page_content=fallback_content,
                        metadata={
                            "source_file": "knowledge_graph.graphml",
                            "doc_type": "knowledge_graph",
                            "entity_name": loc,
                            "chunk_type": "graph_fallback",
                            "display_header": f">>> PROVIDER FALLBACK: {loc} <<<\n"
                        }
                    ))

        return docs

if __name__ == "__main__":
    retriever = GraphRetriever()
    test_query = "What is the copay for metformin on the Silver plan and which doctors in Chicago accept it?"
    results = retriever.invoke(test_query)
    for d in results:
        print(d.page_content)
        print("-" * 40)
