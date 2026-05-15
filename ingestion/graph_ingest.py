"""
Knowledge Graph Ingestion for Health Insurance RAG.

Parses CSV files to build a NetworkX Knowledge Graph of Drugs,
Providers, Conditions, and their relationships.

Usage:
    python -m ingestion.graph_ingest
    # or from project root:
    python ingestion/graph_ingest.py
"""

import sys
import os

# Ensure project root is on sys.path so `config` can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import networkx as nx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from config import (
    DOCUMENTS_DIR,
    GRAPH_DATA_PATH,
)

console = Console()

def build_knowledge_graph():
    """Extract entities and relationships from CSVs and build a NetworkX graph."""
    G = nx.MultiDiGraph()

    drug_file = os.path.join(DOCUMENTS_DIR, "Drug_Formulary_2025.csv")
    provider_file = os.path.join(DOCUMENTS_DIR, "InNetwork_Provider_Directory.csv")

    if not os.path.exists(drug_file) or not os.path.exists(provider_file):
        console.print("[bold red]Error:[/] CSV files not found in data/ directory.")
        return

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        # 1. Process Drug Formulary
        task1 = progress.add_task("Building Drug & Condition nodes...", total=None)
        df_drug = pd.read_csv(drug_file)

        for _, row in df_drug.iterrows():
            drug_name  = str(row['drug_name']).strip()
            brand_name = str(row['brand_name']).strip()
            tier       = str(row['tier']).strip()
            drug_class = str(row['drug_class']).strip()

            # Drug Node
            G.add_node(drug_name, type="Drug", brand=brand_name, drug_class=drug_class)

            # Condition Nodes & TREATS Relationships
            conditions = str(row['covered_conditions']).split(',')
            for cond in conditions:
                cond = cond.strip()
                if cond:
                    G.add_node(cond, type="Condition")
                    G.add_edge(drug_name, cond, relation="TREATS")

            # Plan Tier Relationships
            for tier_name in ["Bronze", "Silver", "Gold"]:
                copay_col = f"{tier_name.lower()}_copay"
                if copay_col in row and pd.notna(row[copay_col]):
                    G.add_node(tier_name, type="PlanTier")
                    G.add_edge(drug_name, tier_name, relation="IN_TIER", copay=str(row[copay_col]))

        progress.update(task1, description="Drug nodes complete!", completed=True)

        # 2. Process Provider Directory
        task2 = progress.add_task("Building Provider & Hospital nodes...", total=None)
        df_prov = pd.read_csv(provider_file)

        for _, row in df_prov.iterrows():
            npi      = str(row['npi']).strip()
            name     = f"{row['provider_first_name']} {row['provider_last_name']}".strip()
            specialty = str(row['specialty']).strip()
            hospital  = str(row['hospital_affiliation']).strip()
            city      = str(row['city']).strip()
            state     = str(row['state']).strip()

            # Provider Node
            G.add_node(npi, type="Provider", name=name, specialty=specialty, city=city, state=state)

            # Specialty Node
            G.add_node(specialty, type="Specialty")
            G.add_edge(npi, specialty, relation="SPECIALIZES_IN")

            # City Node
            city_node = f"{city}, {state}"
            G.add_node(city_node, type="Location", city=city, state=state)
            G.add_edge(npi, city_node, relation="LOCATED_IN")

            # Hospital Node
            if hospital and hospital.lower() != "n/a" and "outpatient only" not in hospital.lower():
                G.add_node(hospital, type="Hospital")
                G.add_edge(npi, hospital, relation="AFFILIATED_WITH")

            # Plan Tier Relationships
            tiers_accepted = str(row['plan_tiers_accepted']).split(',')
            for tier in tiers_accepted:
                tier = tier.strip()
                if tier:
                    G.add_node(tier, type="PlanTier")
                    G.add_edge(npi, tier, relation="ACCEPTS_PLAN")

        progress.update(task2, description="Provider nodes complete!", completed=True)

    # Save the graph
    console.print(f"\n💾 Saving Knowledge Graph to [bold]{GRAPH_DATA_PATH}[/]...")
    nx.write_graphml(G, GRAPH_DATA_PATH)
    console.print(f"✅ Graph built with [bold]{G.number_of_nodes()}[/] nodes and [bold]{G.number_of_edges()}[/] edges.")

if __name__ == "__main__":
    console.print("\n[bold magenta]═══════════════════════════════════════════════════[/]")
    console.print("[bold magenta]   Health Insurance Knowledge Base — Graph Ingest[/]")
    console.print("[bold magenta]═══════════════════════════════════════════════════[/]\n")
    build_knowledge_graph()
