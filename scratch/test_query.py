import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.graph_retriever import GraphRetriever

gr = GraphRetriever()
print("Graph loaded successfully.")
print("Entities:", gr._extract_entities("find heart doctors in silver plan"))
try:
    docs = gr.invoke("find heart doctors in silver plan")
    print(f"Success! Retrieved {len(docs)} documents.")
    for d in docs:
        print("DOCUMENT:")
        print(d.page_content)
        print("-" * 50)
except Exception as e:
    print("CRASHED:", e)
    import traceback
    traceback.print_exc()
