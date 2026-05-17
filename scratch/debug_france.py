from retriever import get_retriever
import sys

def analyze(query):
    retriever = get_retriever()
    print(f"\n--- Analyzing retrieval for query: '{query}' ---")
    docs = retriever.invoke(query)
    
    if docs:
        print("\n--- Full Document Object Inspection ---")
        print(vars(docs[0]))
    
    for i, doc in enumerate(docs):
        print(f"\n[Chunk {i+1}]")
        print(f"Metadata: {doc.metadata}")
        print(f"Content Preview: {doc.page_content[:500]}...")

if __name__ == "__main__":
    analyze("Will you pay less if you use a network provider?")
