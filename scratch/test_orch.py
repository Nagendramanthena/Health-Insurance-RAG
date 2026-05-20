import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from orchestration.orchestrator import Orchestrator

orch = Orchestrator()
print("Orchestrator loaded.")

try:
    res = orch.ask_detailed("find heart doctors in silver plan")
    print("SUCCESS!")
    print("Intent:", res["intent"])
    print("Steps log:")
    for step in res["steps_log"]:
        print(f" - {step}")
    print("Answer:")
    print(res["answer"])
except Exception as e:
    print("CRASHED:", e)
    import traceback
    traceback.print_exc()
