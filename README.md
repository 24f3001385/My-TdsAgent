# Data-Analyst Telegram Bot

An LLM agent, reachable over Telegram, that answers data-analysis questions
(MOSPI and similar public datasets) and replies to every message with
exactly one JSON object:

```json
{"answer": <answer, shaped as the question asks>, "log_url": "https://your-host/run.jsonl"}
```

## Architecture

One process, three things running:

```
FastAPI app ──► GET /health        keep-alive + sanity check
            └─► GET /run.jsonl     public agent log (this is your log_url)

Background thread ──► Telegram getUpdates long-poll loop
                      └─► per message: agent loop → sendMessage(JSON)

Background thread ──► self-pings /health every 10 min (free hosts idle out)
```

Long polling (not a webhook) — works from any host, no HTTPS/webhook setup.

- `bot.py` — starts uvicorn + the two background threads. Replies to
  **every** incoming text message (including multi-turn setup messages),
  always overwrites `log_url` with the real current URL.
- `agent.py` — the agent loop against Google Gemini (`gemini-3.5-flash` by
  default, free tier via the `google-genai` SDK). One tool: `run_python`.
  Wall-clock budget (~210s) forces a final answer before the grader's
  ~300s timeout hits. Robust JSON extraction: strips fences, finds the
  first balanced `{...}`, wraps bare values under `"answer"` if needed.
  Note: free/smaller models can get obscure real-world statistics wrong
  (see the grading guide's test table) — lean on `run_python` actually
  fetching real data rather than trusting the model's own knowledge.
- `tools/exec_pandas.py` — sandboxed subprocess execution with pandas,
  numpy, requests, BeautifulSoup pre-imported.
- `logger.py` — appends one JSON line per step (question, every tool call +
  result, final answer, errors) to `run.jsonl` at the project root.

## Local setup

```bash
git clone <this-repo>
cd telegram-data-analyst-bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in real values
python bot.py
```

Required env vars (see `.env.example`):
- `BOT_TOKEN` — from @BotFather (`/newbot`; username must end in `bot`)
- `GEMINI_API_KEY` — free key from https://aistudio.google.com/apikey
- `GEMINI_MODEL` — defaults to `gemini-3.5-flash`
- `BASE_URL` — your deployed HTTPS base URL (used for `log_url` and the
  self-ping)

No webhook registration step needed — long polling starts as soon as
`bot.py` runs.

## Deploy (Render)

1. Push this repo to GitHub (public, no secrets committed).
2. Render → New → Web Service → connect the repo. `render.yaml` sets:
   build `pip install -r requirements.txt`, start `python bot.py`.
3. Set the env vars above in the Render dashboard.
4. **Changing env vars on Render does not restart the service** — trigger
   a manual deploy afterwards.
5. Verify:
   ```bash
   curl https://your-app.onrender.com/health
   wget https://your-app.onrender.com/run.jsonl
   ```

The self-ping thread keeps the free instance awake; an external pinger
(UptimeRobot) is a good belt-and-suspenders backup.

## Testing against the grading harness

```bash
git clone https://github.com/Jivraj-18/tds-p1-t2-2026-telegram-bot
cd tds-p1-t2-2026-telegram-bot
# point it at your bot username; add your own questions to
# evals/questions.json for a full dress rehearsal
```

Also test by hand:
- Message your bot directly from your own Telegram account, check you get
  one clean JSON object back, nothing else.
- Test a multi-turn flow: send `"I will send data next."`, then the real
  question — the bot must reply to **both** messages.
- `wget` your `log_url` from a different network to confirm it's truly
  public and shows the run you just did.

## Checklist before you walk away

- [ x] Bot replies to a fresh message with exactly one JSON object
- [x ] `answer` shape matches whatever the message asked for
- [ x] `log_url` is wget-able and reflects the run you just did
- [ x] Multi-turn: bot replies to every message, not just the last
- [ x] Reply arrives well under 300s even on a hard question
- [x ] Repo is public; no secrets committed (tokens only in env vars)
- [ x] Host stays awake (self-ping working)
- [x] `GEMINI_API_KEY` will still be valid weeks from now (check free-tier quota)
- [x] Registered on SEEK: `https://github.com/you/repo, your_bot_username`
