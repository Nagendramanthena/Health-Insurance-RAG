import contextvars

# ContextVar to store a list of trace logs for the current request thread
trace_log = contextvars.ContextVar("trace_log", default=None)

def log_event(msg: str):
    """Log an event to the current context's trace log."""
    log = trace_log.get()
    if log is not None:
        log.append(msg)
