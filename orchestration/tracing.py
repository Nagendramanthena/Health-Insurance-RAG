import contextvars
import concurrent.futures

# ContextVar to store a list of trace logs for the current request thread
trace_log = contextvars.ContextVar("trace_log", default=None)

def log_event(msg: str):
    """Log an event to the current context's trace log."""
    log = trace_log.get()
    if log is not None:
        log.append(msg)
    else:
        print(f"TRACE_LOG IS NONE FOR MSG: {msg}", flush=True)

# Global Monkey Patch: Ensure all ThreadPoolExecutor threads inherit contextvars
_original_submit = concurrent.futures.ThreadPoolExecutor.submit

def _patched_submit(self, fn, *args, **kwargs):
    ctx = contextvars.copy_context()
    def _wrapper(*wargs, **wkwargs):
        return ctx.run(fn, *wargs, **wkwargs)
    return _original_submit(self, _wrapper, *args, **kwargs)

concurrent.futures.ThreadPoolExecutor.submit = _patched_submit

