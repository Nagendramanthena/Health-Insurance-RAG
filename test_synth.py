from orchestration.orchestrator import Orchestrator

orch = Orchestrator()
# Just run the full query
res = orch.ask_detailed("Hello, can you find heart doctors under silver plan")
print(res["answer"])
