from orchestration.orchestrator import Orchestrator
orch = Orchestrator()
res = orch.ask_detailed("Hello, can you find heart doctors under silver plan")
print(res["steps_log"])
print("ANSWER:")
print(res["answer"])
