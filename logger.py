"""
Append-only JSONL logger for agent runs.

Every step of every run is written as one JSON line to run.jsonl at the
project root. bot.py serves this file at GET /run.jsonl (per the grading
guide's expected route), so the same URL is always current — that's the
log_url you report to the grader.
"""

import json
import os
import threading
import time
import uuid

LOG_PATH = os.path.join(os.path.dirname(__file__), "run.jsonl")

_lock = threading.Lock()


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def log_event(run_id: str, chat_id, event: str, data: dict):
    """Append one structured event to the JSONL log."""
    record = {
        "ts": time.time(),
        "run_id": run_id,
        "chat_id": chat_id,
        "event": event,
        **data,
    }
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def log_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/run.jsonl"
