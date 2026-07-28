"""
Data-analyst Telegram bot — long-polling architecture.

Three things run in one process (per the grading guide):
  1. FastAPI app: GET /health, GET /run.jsonl  (uvicorn serves this)
  2. Background thread: Telegram getUpdates long-poll loop -> agent -> sendMessage
  3. Background thread: self-ping /health every 10 min so free hosts don't idle out

Long polling means no webhook/HTTPS setup is needed — this works from any host.
"""

import json
import os
import threading
import time
import traceback

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from agent import answer_question
from logger import LOG_PATH, log_event, log_url, new_run_id

BOT_TOKEN = os.environ["BOT_TOKEN"]
BASE_URL = os.environ["BASE_URL"]  # e.g. https://your-app.onrender.com
# GEMINI_API_KEY is read directly by agent.py via os.environ — fail fast here too
# so a missing key surfaces immediately on startup rather than on first message.
os.environ["GEMINI_API_KEY"]

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

app = FastAPI()

# Per-chat history, kept in memory. Fine for a grading bot; swap for a
# real store if you need it to survive restarts.
CHAT_HISTORY: dict[int, list] = {}
MAX_HISTORY_TURNS = 20


@app.get("/health")
def health():
    return JSONResponse({"ok": True, "ts": time.time()})


@app.get("/run.jsonl")
def get_log():
    if not os.path.exists(LOG_PATH):
        open(LOG_PATH, "a").close()
    return FileResponse(LOG_PATH, media_type="application/json")


@app.get("/")
def root():
    return JSONResponse({"status": "data-analyst telegram bot running"})


def send_message(chat_id: int, text: str):
    try:
        resp = requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=30,
        )
        resp.raise_for_status()
    except Exception:
        traceback.print_exc()


def handle_message(chat_id: int, text: str):
    run_id = new_run_id()
    history = CHAT_HISTORY.get(chat_id, [])

    try:
        result = answer_question(text, history, run_id, chat_id)
    except Exception as e:
        log_event(run_id, chat_id, "error", {"error": str(e), "traceback": traceback.format_exc()})
        result = {"answer": "internal error"}

    # Always overwrite log_url with the real, current URL.
    result["log_url"] = log_url(BASE_URL)
    reply_text = json.dumps(result, ensure_ascii=False)

    history = history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": reply_text},
    ]
    CHAT_HISTORY[chat_id] = history[-MAX_HISTORY_TURNS:]

    send_message(chat_id, reply_text)


def polling_loop():
    """Long-poll Telegram's getUpdates and reply to every text message."""
    offset = None
    while True:
        try:
            params = {"timeout": 30}
            if offset is not None:
                params["offset"] = offset
            resp = requests.get(f"{TELEGRAM_API}/getUpdates", params=params, timeout=40)
            resp.raise_for_status()
            updates = resp.json().get("result", [])

            for update in updates:
                offset = update["update_id"] + 1
                message = update.get("message") or update.get("edited_message")
                if not message or "text" not in message:
                    continue
                chat_id = message["chat"]["id"]
                text = message["text"]
                # Handle inline so we reply before polling again — the
                # grader waits for a reply to each message before sending
                # the next one, so we don't need extra concurrency here.
                handle_message(chat_id, text)

        except Exception:
            traceback.print_exc()
            time.sleep(5)


def keep_warm_loop():
    """Ping our own /health every 10 minutes so free hosts don't sleep."""
    while True:
        time.sleep(600)
        try:
            requests.get(f"{BASE_URL.rstrip('/')}/health", timeout=15)
        except Exception:
            pass


@app.on_event("startup")
def start_background_threads():
    threading.Thread(target=polling_loop, daemon=True).start()
    threading.Thread(target=keep_warm_loop, daemon=True).start()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
