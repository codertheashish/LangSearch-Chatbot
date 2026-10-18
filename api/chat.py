import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote

import requests
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("langsearch")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Current production model per https://console.groq.com/docs/models.
# llama-3.3-70b-versatile was retired for free/developer tiers on 2026-08-16.
# Override without code changes via the GROQ_MODEL environment variable.
DEFAULT_MODEL = "openai/gpt-oss-120b"

HEADERS = {"User-Agent": "LangSearchChatbot/1.0 (vercel deployment)"}

MAX_HISTORY = 12            # messages kept from the browser
MAX_MESSAGE_CHARS = 4000    # per message
MAX_TOOL_ROUNDS = 4         # Wikipedia/ArXiv rounds before forcing an answer
TIME_BUDGET_SECONDS = 50    # must stay below maxDuration (60s) in vercel.json

SYSTEM_PROMPT = (
    "You are LangSearch, a helpful assistant that can search Wikipedia and "
    "ArXiv. Use the tools when you need facts or research papers, then answer "
    "clearly and concisely. If a tool fails, say so and answer with what you know."
)


# ---------------- Tools ----------------

def wikipedia_search(query: str) -> str:
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query,
                    "format": "json", "srlimit": 1},
            headers=HEADERS, timeout=8,
        )
        r.raise_for_status()
        hits = r.json().get("query", {}).get("search", [])
        if not hits:
            return "No Wikipedia results found."
        title = hits[0]["title"]
        s = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(title.replace(" ", "_")),
            headers=HEADERS, timeout=8,
        )
        s.raise_for_status()
        extract = s.json().get("extract", "")
        return f"Page: {title}\nSummary: {extract[:1000]}"
    except Exception as e:
        log.warning("Wikipedia tool failed: %s", type(e).__name__)
        return f"Wikipedia search failed: {type(e).__name__}. Try rephrasing the query."


def arxiv_search(query: str) -> str:
    try:
        r = requests.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": f"all:{query}", "max_results": 2},
            headers=HEADERS, timeout=10,
        )
        r.raise_for_status()
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entries = ET.fromstring(r.text).findall("a:entry", ns)
        if not entries:
            return "No ArXiv papers found."
        out = []
        for e in entries:
            title = " ".join(e.findtext("a:title", "", ns).split())
            summary = " ".join(e.findtext("a:summary", "", ns).split())[:500]
            link = e.findtext("a:id", "", ns)
            out.append(f"Title: {title}\nLink: {link}\nSummary: {summary}")
        return "\n\n".join(out)
    except Exception as e:
        log.warning("ArXiv tool failed: %s", type(e).__name__)
        return f"ArXiv is unavailable right now (possibly rate-limited): {type(e).__name__}"


TOOL_FUNCS = {"wikipedia": wikipedia_search, "arxiv": arxiv_search}

TOOLS = [
    {"type": "function", "function": {
        "name": "wikipedia",
        "description": "Search Wikipedia for general knowledge and definitions.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "arxiv",
        "description": "Search ArXiv.org for scientific papers.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
]


# ---------------- Groq call ----------------

class GroqError(Exception):
    """Upstream (Groq) failure. `detail` is for server logs only."""

    def __init__(self, status, detail, code=None):
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.code = code


def groq_chat(api_key, model, messages, use_tools, timeout):
    body = {"model": model, "messages": messages, "temperature": 0.3}
    if use_tools:
        body["tools"] = TOOLS
        body["tool_choice"] = "auto"
    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body, timeout=timeout,
        )
    except requests.Timeout:
        raise GroqError(504, "Groq request timed out")
    except requests.RequestException as e:
        raise GroqError(502, f"Could not reach Groq: {type(e).__name__}")

    if r.status_code != 200:
        detail, code = r.text[:500], None
        try:
            err = r.json().get("error", {})
            detail = err.get("message", detail)
            code = err.get("code")
        except Exception:
            pass
        raise GroqError(r.status_code, str(detail).replace(api_key, "[redacted]"), code)

    try:
        message = r.json()["choices"][0]["message"]
        if not isinstance(message, dict):
            raise TypeError("message is not an object")
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise GroqError(502, f"Unexpected response format from Groq: {type(e).__name__}")
    return message


def public_message(err: GroqError):
    """Map an upstream error to (safe text for the browser, HTTP status)."""
    if err.status in (401, 403):
        return "The AI service rejected this server's credentials. The site owner needs to check the API key.", 502
    if err.status == 404:
        return "The AI model is currently unavailable. The site owner needs to update the model setting.", 502
    if err.status == 429:
        return "Too many requests right now. Please wait a moment and try again.", 429
    if err.status == 504:
        return "The AI service took too long to respond. Please try again.", 504
    return "The AI service is temporarily unavailable. Please try again shortly.", 502


# ---------------- Routes ----------------

def _config():
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    model = os.environ.get("GROQ_MODEL", "").strip() or DEFAULT_MODEL
    return api_key, model


def _clean_history(raw):
    if not isinstance(raw, list):
        return None
    cleaned = []
    for m in raw[-MAX_HISTORY:]:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content[:MAX_MESSAGE_CHARS]})
    if not cleaned or cleaned[-1]["role"] != "user":
        return None
    return cleaned


@app.route("/api/chat", methods=["GET", "POST"])
def chat():
    api_key, model = _config()

    if request.method == "GET":  # health check: never reveals the key
        return jsonify(status="ok", configured=bool(api_key), model=model)

    if not api_key:
        log.error("GROQ_API_KEY is not set in the server environment")
        return jsonify(error="The chatbot is not configured yet. Please try again later."), 503

    data = request.get_json(silent=True)
    history = _clean_history(data.get("messages") if isinstance(data, dict) else None)
    if history is None:
        return jsonify(error="Please send a message and try again."), 400

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    deadline = time.monotonic() + TIME_BUDGET_SECONDS
    tools_enabled, tool_rounds = True, 0

    try:
        for _ in range(MAX_TOOL_ROUNDS + 2):
            remaining = deadline - time.monotonic()
            if remaining < 5:
                raise GroqError(504, "Time budget exhausted")

            use_tools = tools_enabled and tool_rounds < MAX_TOOL_ROUNDS
            try:
                msg = groq_chat(api_key, model, messages, use_tools, timeout=min(30, remaining))
            except GroqError as e:
                if e.status == 400 and e.code == "tool_use_failed" and tools_enabled:
                    log.warning("Model produced an invalid tool call; retrying without tools")
                    tools_enabled = False
                    continue
                raise

            calls = msg.get("tool_calls")
            if not calls:
                reply = (msg.get("content") or "").strip()
                if not reply:
                    raise GroqError(502, "Model returned an empty answer")
                return jsonify(reply=reply)

            tool_rounds += 1
            messages.append({"role": "assistant", "content": msg.get("content") or "",
                             "tool_calls": calls})
            for c in calls:
                fn_info = c.get("function") or {}
                name = str(fn_info.get("name", "")).split(".")[-1]
                try:
                    args = json.loads(fn_info.get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                query = str(args.get("query", ""))[:300] if isinstance(args, dict) else ""
                fn = TOOL_FUNCS.get(name)
                result = fn(query) if fn and query else "Unknown tool or empty query."
                messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": result})

        raise GroqError(502, "No final answer after tool rounds")

    except GroqError as e:
        log.error("Groq failure: status=%s code=%s model=%s detail=%s", e.status, e.code, model, e.detail)
        text, status = public_message(e)
        return jsonify(error=text), status


@app.errorhandler(HTTPException)
def http_error(e):
    return jsonify(error=e.description or "Request failed."), e.code


@app.errorhandler(Exception)
def unexpected_error(e):
    log.exception("Unhandled error in /api/chat")
    return jsonify(error="Something went wrong on the server. Please try again."), 500


@app.after_request
def no_store(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp
