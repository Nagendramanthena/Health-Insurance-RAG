import os
import sys
from orchestration.tools import relational_search

query = "i need doctors - Hello ,can you find heart doctors under silver plan"
result = relational_search.invoke({"query": query})
print(result)
