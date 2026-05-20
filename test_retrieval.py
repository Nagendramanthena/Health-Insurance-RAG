import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from orchestration.tools import plan_comparison_search
query = "What is the deductible for the Bronze plan compared to the Gold plan?"
for tier in ["Bronze", "Gold"]:
    print(f"=== {tier} ===")
    res = plan_comparison_search.invoke({"query": query, "tier": tier})
    print(res[:500])
