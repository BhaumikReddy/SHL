"""
app/agent.py

Agent logic: builds a query from conversation history, retrieves relevant
catalog items via FAISS, calls Gemini (gemini-2.0-flash), and parses
the JSON response.
"""

import json
import os
import re

from dotenv import load_dotenv
from google import genai

from . import retriever

load_dotenv()

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is missing. "
                "Please set it in your .env file."
            )
        _client = genai.Client(api_key=key)
    return _client


_MODEL = "gemini-2.0-flash"

MAX_TURNS = 8

KEYS_TO_TYPE = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


def get_test_type(item: dict) -> str:
    codes = [KEYS_TO_TYPE.get(k, "") for k in item.get("keys", []) if k in KEYS_TO_TYPE]
    return ",".join(filter(None, codes))


SYSTEM_PROMPT = """\
You are an SHL assessment recommender. Your job is to help hiring managers find \
the right SHL assessments from the official catalog.

Rules:
- Only recommend assessments from the AVAILABLE ASSESSMENTS list provided below.
- Never invent assessment names or URLs.
- If the user's request is too vague (no role, no level, no skill to measure), ask ONE clarifying question.
- When you have enough context always recommend between 5 and 10 assessments, not fewer.
- Use the following fields to match assessments to the user's requirements:
    * job_levels — match seniority (e.g. Director, Graduate, Entry-Level, Manager)
    * duration — mention timing when the user asks about it
    * languages — filter if the user specifies language requirements
    * remote — mention if the user asks about remote-friendly testing
    * adaptive — mention if relevant
    * keys — use for test category matching (e.g. Personality & Behavior, Ability & Aptitude)
- If the user asks to compare assessments, compare only using the catalog data given.
- Refuse off-topic questions (legal advice, general HR, anything unrelated to SHL assessments).
- Do not recommend on the very first turn if the query is vague.

Response format — ALWAYS respond with ONLY valid JSON, no extra text:

When clarifying:
{"reply": "Your clarifying question here.", "recommendations": [], "end_of_conversation": false}

When recommending:
{"reply": "Brief explanation.", "recommendations": [{"name": "...", "url": "...", "test_type": "..."}], "end_of_conversation": false}

When the user is satisfied:
{"reply": "...", "recommendations": [...], "end_of_conversation": true}

Important:
- "url" must be copied exactly from the catalog entry.
- "test_type" must be the single-letter code(s) (e.g. "K", "A,S", "P").
- Always return valid JSON. Never return plain text outside of JSON.
"""

FALLBACK = {"reply": "", "recommendations": [], "end_of_conversation": False}


def _build_query(messages: list[dict]) -> str:
    user_texts = [m["content"] for m in messages if m["role"] == "user"]
    return " ".join(user_texts[-3:])


def _format_catalog_context(items: list[dict]) -> str:
    lines = []
    for item in items:
        keys = ", ".join(item.get("keys", []))
        test_type = get_test_type(item)
        levels = ", ".join(item.get("job_levels", []))
        langs = ", ".join(item.get("languages", [])[:5])
        duration = item.get("duration", "—")
        remote = item.get("remote", "—")
        adaptive = item.get("adaptive", "—")
        desc = (item.get("description", "") or "")[:200]

        lines.append(
            f"- Name: {item['name']}\n"
            f"  URL: {item['link']}\n"
            f"  Test Type Code: {test_type}\n"
            f"  Keys: {keys}\n"
            f"  Job Levels: {levels}\n"
            f"  Duration: {duration}\n"
            f"  Languages: {langs}\n"
            f"  Remote: {remote} | Adaptive: {adaptive}\n"
            f"  Description: {desc}"
        )
    return "\n\n".join(lines)


def _build_prompt(messages: list[dict], system_with_catalog: str) -> str:
    """
    Build a single prompt string combining system instructions,
    catalog context, and full conversation history.
    """
    prompt = system_with_catalog + "\n\n"
    prompt += "CONVERSATION HISTORY:\n"
    for m in messages:
        role = "User" if m["role"] == "user" else "Assistant"
        prompt += f"{role}: {m['content']}\n"
    prompt += "\nAssistant:"
    return prompt


def _parse_response(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {**FALLBACK, "reply": text}

    try:
        data = json.loads(match.group())
        reply = str(data.get("reply", ""))
        recs = data.get("recommendations", [])
        end = bool(data.get("end_of_conversation", False))

        clean_recs = []
        for r in recs[:10]:
            if isinstance(r, dict) and r.get("name") and r.get("url"):
                clean_recs.append({
                    "name": r.get("name", ""),
                    "url": r.get("url", ""),
                    "test_type": r.get("test_type", ""),
                })

        return {"reply": reply, "recommendations": clean_recs, "end_of_conversation": end}

    except (json.JSONDecodeError, ValueError):
        return {**FALLBACK, "reply": text}


def get_agent_reply(messages: list[dict]) -> dict:
    """
    Main agent entry point using Gemini via google-genai SDK.
    """
    user_turns = sum(1 for m in messages if m["role"] == "user")
    if user_turns > MAX_TURNS:
        return {
            "reply": "We've reached the maximum conversation length. Please start a new session.",
            "recommendations": [],
            "end_of_conversation": True,
        }

    query = _build_query(messages)
    catalog_items = retriever.search(query, top_k=20)

    catalog_context = _format_catalog_context(catalog_items)
    system_with_catalog = (
        SYSTEM_PROMPT
        + f"\n\nAVAILABLE ASSESSMENTS (use only these):\n\n{catalog_context}"
    )

    prompt = _build_prompt(messages, system_with_catalog)

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=_MODEL,
            contents=prompt,
        )
        raw_text = response.text
    except Exception as e:
        return {
            **FALLBACK,
            "reply": f"Sorry, I encountered an error: {str(e)}. Please try again.",
        }

    return _parse_response(raw_text)