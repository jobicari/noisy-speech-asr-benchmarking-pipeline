import io
import json
import contextlib


def capture_timing(fn, *args, **kwargs):
    """Run fn and return (its result, the [TIMING] records it printed)"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    records = [json.loads(line[len("[TIMING] "):]) for line in buf.getvalue().splitlines()
               if line.startswith("[TIMING] ")]
    return result, records


def events(records, event, name=None):
    return [r for r in records if r["event"] == event and (name is None or r.get("name") == name)]
