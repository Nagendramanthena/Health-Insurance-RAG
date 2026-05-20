from orchestration.memory import create_session_memory, SESSION_USER
mem = create_session_memory()
messages = [
    {"role": "user",      "content": "i am under silver plan"},
    {"role": "assistant", "content": "Got it, I will remember that you are on the Silver Plan."},
]
res = mem.add(messages, user_id=SESSION_USER)
print("MEM0 RESULT:", res)
