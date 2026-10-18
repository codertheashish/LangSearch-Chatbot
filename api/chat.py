import json
import os
import xml.etree.ElementTree as ET
from urllib.parse import quote

import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"
HEADERS = {"User-Agent": "LangSearchChatbot/1.0 (vercel deployment)"}

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
        hits = r.json().get("query", {}).get("search", [])
        if not hits:
            return "No Wikipedia results found."
        title = hits[0]["title"]
        s = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(title.replace(" ", "_")),
            headers=HEADERS, timeout=8,
        )
        extract = s.json().get("extract", "")
        return f"Page: {title}\nSummary: {extract[:1000]}"
    except Exception as e:
        return f"Wikipedia search failed: {e}. Try rephrasing the query."


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
        return f"ArXiv is unavailable right now (possibly rate-limited): {e}"


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

def groq(api_key, messages, use_tools=True):
    body = {"model": MODEL, "messages": messages, "temperature": 0.3}
    if use_tools:
        body["tools"] = TOOLS
        body["tool_choice"] = "auto"
    r = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body, timeout=40,
    )
    if r.status_code != 200:
        try:
            detail = r.json().get("error", {}).get("message", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"Groq API error ({r.status_code}): {detail}")
    return r.json()["choices"][0]["message"]


# ---------------- Route ----------------

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    api_key = (data.get("api_key") or os.environ.get("GROQ_API_KEY") or "").strip()
    history = data.get("messages") or []

    if not api_key:
        return jsonify(error="Add your Groq API key in Settings."), 400
    if not history:
        return jsonify(error="No message received."), 400

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history[-12:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": str(m["content"])})

    try:
        for _ in range(4):
            msg = groq(api_key, messages)
            calls = msg.get("tool_calls")
            if not calls:
                return jsonify(reply=msg.get("content") or "")
            messages.append({"role": "assistant", "content": msg.get("content") or "",
                             "tool_calls": calls})
            for c in calls:
                name = c["function"]["name"]
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                fn = TOOL_FUNCS.get(name)
                result = fn(args.get("query", "")) if fn else "Unknown tool."
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
        # Out of iterations: force a final answer without tools
        final = groq(api_key, messages, use_tools=False)
        return jsonify(reply=final.get("content") or "")
    except Exception as e:
        return jsonify(error=str(e)), 502
