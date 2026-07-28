"""
The agent: takes a data-analysis question (plus prior turns for context),
works the answer out using the run_python tool, and returns a JSON object
containing at least an "answer" key. bot.py overwrites/adds "log_url"
before sending the reply, so the model never needs to know the real URL.

Defensive layers (per the grading guide):
  - Wall-clock deadline: past ~210s we stop giving the model new tool
    turns and force a final answer, so a late "perfect" answer never
    times out the whole question.
  - JSON extraction: strip fences, find the first balanced {...}, parse
    it. If there's no "answer" key, wrap the whole thing as one.
  - Never crash silently: any failure still returns a parseable dict.
"""

import json
import os
import re
import time

from openai import OpenAI

from logger import log_event
from tools.python_exec import run_python, RUN_PYTHON_TOOL_DEF

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
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

TOOLS = [RUN_PYTHON_TOOL_DEF]


def _extract_json(text: str):
    """Strip fences, find the first balanced {...}, parse it."""
    text = text.strip()
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


def answer_question(question: str, history: list, run_id: str, chat_id) -> dict:
    """
    history: list of {"role": "user"/"assistant", "content": str} prior turns.
    Returns a dict that will have "answer" (and we add "log_url" in bot.py).
    Always returns something parseable — never raises.
    """
    client = OpenAI()  # reads OPENAI_API_KEY from env
    deadline = time.time() + WALL_CLOCK_BUDGET_SECONDS

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})

    log_event(run_id, chat_id, "question_received", {"question": question})

    last_text = ""
    try:
        for iteration in range(MAX_TOOL_ITERATIONS):
            past_deadline = time.time() > deadline
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=None if past_deadline else TOOLS,
                tool_choice="none" if past_deadline else "auto",
            )
            choice = resp.choices[0]
            msg = choice.message

            log_event(
                run_id,
                chat_id,
                "model_response",
                {
                    "iteration": iteration,
                    "past_deadline": past_deadline,
                    "content": msg.content,
                    "tool_calls": [tc.model_dump() for tc in (msg.tool_calls or [])],
                },
            )

            last_text = msg.content or ""

            if not msg.tool_calls:
                break

            messages.append(msg.model_dump(exclude_unset=True))
            for tc in msg.tool_calls:
                if tc.function.name == "run_python":
                    try:
                        args = json.loads(tc.function.arguments)
                        code = args.get("code", "")
                    except json.JSONDecodeError:
                        code = ""
                    result = run_python(code)
                    log_event(
                        run_id,
                        chat_id,
                        "tool_call",
                        {"tool": "run_python", "code": code, "result": result},
                    )
                else:
                    result = f"Unknown tool: {tc.function.name}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )
    except Exception as e:
        log_event(run_id, chat_id, "error", {"error": str(e)})

    parsed = _extract_json(last_text)
    if parsed is None:
        # Last resort: still return something parseable.
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
