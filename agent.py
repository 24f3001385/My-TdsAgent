"""
The agent: takes a data-analysis question (plus prior turns for context),
works the answer out using the run_python tool, and returns a JSON object
containing at least an "answer" key. bot.py overwrites/adds "log_url"
before sending the reply, so the model never needs to know the real URL.

Uses Google's free-tier Gemini API via the `google-genai` SDK (the current,
supported package -- NOT the deprecated `google-generativeai`).

Defensive layers (per the grading guide):
  - Wall-clock deadline: past ~210s we stop giving the model new tool
    turns and force a final answer, so a late "perfect" answer never
    times out the whole question.
  - Retry on transient errors: Gemini's free tier occasionally returns
    503 (overloaded) or 429 (rate-limited); these are retried with
    exponential backoff instead of failing the whole run immediately.
  - JSON extraction: strip fences, find the first balanced {...}, parse
    it. If there's no "answer" key, wrap the whole thing as one.
  - Never crash silently: any failure still returns a parseable dict.
"""

import json
import os
import re
import time

import google.genai as genai
from google.genai import types

from logger import log_event
from tools.python_exec import run_python

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
MAX_TOOL_ITERATIONS = 10
WALL_CLOCK_BUDGET_SECONDS = 210  # stay well under the grader's ~300s timeout

SYSTEM_PROMPT = """You are a rigorous data-analyst agent operating over Telegram.

You will be given a conversation. Answer the LATEST user message; earlier
messages are context for multi-turn questions. The latest message may embed
data inline, or point at a public dataset (MOSPI, data.gov.in, RBI, etc.),
and it will specify the exact JSON shape its "answer" field must take.

Rules:
- Never guess a number you could compute. Use the run_python tool
  (pandas/numpy/requests/BeautifulSoup are pre-imported) to fetch the real
  dataset and compute the actual answer.
- If fetching a dataset fails after a couple of attempts, or the question
  is really just a general-knowledge published statistic, answer from your
  own knowledge rather than stalling — a plausible answer beats a timeout.
- If the latest message is only a setup message (e.g. "I will send data
  next.", with no actual question yet), you MUST still reply — respond
  with a small JSON acknowledgement such as {"answer": "ack"}. Never skip
  replying to a message.
- Output ONLY the JSON object for your final answer — no prose before or
  after, no markdown code fences, no explanation. Match the requested
  shape exactly (same keys, same nesting, number vs string as asked).
  Do not include a "log_url" key — that gets added automatically.

Example: if the question asks for {"answer": {"state": "<state name>"}, ...},
your entire final reply should be exactly:
{"answer": {"state": "Assam"}}
"""

RUN_PYTHON_DECLARATION = types.FunctionDeclaration(
    name="run_python",
    description=(
        "Execute a Python snippet to fetch and compute over real data "
        "(pandas, numpy, requests, BeautifulSoup are pre-imported). "
        "Always print() the result you need to see — only stdout/stderr "
        "is returned. Never guess a number you could compute here."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. Use print() to output results.",
            }
        },
        "required": ["code"],
    },
)

TOOLS = [types.Tool(function_declarations=[RUN_PYTHON_DECLARATION])]


def _extract_json(text: str):
    """Strip fences, find the first balanced {...}, parse it."""
    text = (text or "").strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def _history_to_contents(history: list, question: str) -> list:
    """Convert plain {"role","content"} turns into Gemini's Content format.
    Gemini uses "model" instead of "assistant" for the model's turns.
    """
    contents = []
    for turn in history:
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=turn["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
    return contents


_TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "overloaded", "high demand")


def _is_transient(err: Exception) -> bool:
    msg = str(err)
    return any(marker in msg for marker in _TRANSIENT_MARKERS)


def _generate_with_retry(client, contents, config, run_id, chat_id, deadline, max_attempts=5):
    """
    Calls generate_content, retrying with exponential backoff on transient
    errors (503 overloaded, 429 rate-limited) since Gemini's free tier gets
    bursty under load. Gives up and returns None if we run past the wall-clock
    deadline or exhaust max_attempts, so a bad run still ends in a final
    answer instead of hanging until the grader's own timeout.
    """
    delay = 2
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        try:
            return client.models.generate_content(model=MODEL, contents=contents, config=config)
        except Exception as e:
            if not _is_transient(e) or attempt == max_attempts - 1:
                raise
            log_event(
                run_id,
                chat_id,
                "retry",
                {"attempt": attempt, "error": str(e), "sleeping_seconds": delay},
            )
            time.sleep(min(delay, max(0, deadline - time.time())))
            delay *= 2
    return None


def answer_question(question: str, history: list, run_id: str, chat_id) -> dict:
    """
    history: list of {"role": "user"/"assistant", "content": str} prior turns.
    Returns a dict that will have "answer" (and we add "log_url" in bot.py).
    Always returns something parseable — never raises.
    """
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    deadline = time.time() + WALL_CLOCK_BUDGET_SECONDS

    contents = _history_to_contents(history, question)

    log_event(run_id, chat_id, "question_received", {"question": question})

    last_text = ""
    try:
        for iteration in range(MAX_TOOL_ITERATIONS):
            past_deadline = time.time() > deadline
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=None if past_deadline else TOOLS,
            )
            resp = _generate_with_retry(client, contents, config, run_id, chat_id, deadline)
            if resp is None:
                # Exhausted retries with time still left, or ran out of time mid-retry.
                break

            candidate = resp.candidates[0]
            parts = candidate.content.parts or []
            function_calls = [p.function_call for p in parts if p.function_call]
            text_parts = [p.text for p in parts if p.text]
            last_text = "\n".join(text_parts)

            log_event(
                run_id,
                chat_id,
                "model_response",
                {
                    "iteration": iteration,
                    "past_deadline": past_deadline,
                    "text": last_text,
                    "function_calls": [
                        {"name": fc.name, "args": dict(fc.args)} for fc in function_calls
                    ],
                },
            )

            if not function_calls:
                break

            # Append the model's turn (including its function call parts).
            contents.append(candidate.content)

            response_parts = []
            for fc in function_calls:
                if fc.name == "run_python":
                    code = fc.args.get("code", "")
                    result = run_python(code)
                    log_event(
                        run_id,
                        chat_id,
                        "tool_call",
                        {"tool": "run_python", "code": code, "result": result},
                    )
                else:
                    result = f"Unknown tool: {fc.name}"

                response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=fc.name,
                            response={"output": result},
                        )
                    )
                )

            contents.append(types.Content(role="user", parts=response_parts))
    except Exception as e:
        log_event(run_id, chat_id, "error", {"error": str(e)})

    parsed = _extract_json(last_text)
    if parsed is None:
        parsed = {"answer": last_text.strip() or "internal error"}
    elif "answer" not in parsed:
        parsed = {"answer": parsed}

    log_event(
        run_id,
        chat_id,
        "final_answer",
        {"raw_text": last_text, "parsed": parsed},
    )
    return parsed