"""
Runs short data-analysis Python snippets (pandas/numpy/requests/BeautifulSoup)
in a subprocess with a timeout, and returns captured stdout/stderr.

This is the ONLY tool the agent gets. The model must call it to fetch and
compute over real data instead of guessing statistics from memory.
"""

import subprocess
import sys
import tempfile
import textwrap

TIMEOUT_SECONDS = 45

PRELUDE = textwrap.dedent(
    """
    import pandas as pd
    import numpy as np
    import requests
    import io, re, json, math, datetime
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        pass
    """
)


def run_python(code: str) -> str:
    """
    Executes `code` in a fresh python3 subprocess.
    The snippet should print() whatever it wants the agent to see.
    Returns combined stdout+stderr (truncated) as a string.
    """
    full_code = PRELUDE + "\n" + code

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        out = proc.stdout
        err = proc.stderr
        result = ""
        if out:
            result += out
        if err:
            result += f"\n[stderr]\n{err}"
        if not result.strip():
            result = "(no output — did you forget to print()?)"
        return result[:8000]
    except subprocess.TimeoutExpired:
        return f"Execution timed out after {TIMEOUT_SECONDS}s."
    except Exception as e:
        return f"Execution error: {e}"


# OpenAI function-calling tool schema (Chat Completions "tools" format).
RUN_PYTHON_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Execute a Python snippet to fetch and compute over real data "
            "(pandas, numpy, requests, BeautifulSoup are pre-imported). "
            "Always print() the result you need to see — only stdout/stderr "
            "is returned. Never guess a number you could compute here."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Use print() to output results.",
                }
            },
            "required": ["code"],
        },
    },
}
