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
import google.generativeai as genai

from . import retriever

load_dotenv()

_client = None


def _get_client():
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY environment variable is missing. Please set it in your .env file.")
        genai.configure(api_key=key)
        _client = genai.GenerativeModel("gemini-2.0-flash")
    return _client


_MODEL = "gemini-2.0-flash"

MAX_TURNS = 8  # hard cap on conversation length

# Mapping from full key name to single-letter test type code
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
    """Return comma-joined single-letter codes for all keys of an item."""
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
    * adaptive — mention if relevant (adaptive tests adjust to candidate ability)
    * keys — use for test category matching (e.g. Personality & Behavior, Ability & Aptitude, Knowledge & Skills)
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

Important notes on fields:
- "url" in recommendations must be copied exactly from the URL field of the matching catalog entry.
- "test_type" must be the single-letter code(s) for that assessment (e.g. "K", "A", "P", "K,A").
- Always return valid JSON. Never return plain text outside of JSON.
"""

FALLBACK = {"reply": "", "recommendations": [], "end_of_conversation": False}


def _build_query(messages: list[dict]) -> str:
    """Extract a search query from the conversation — combine recent user turns."""
    user_texts = [m["content"] for m in messages if m["role"] == "user"]
    return " ".join(user_texts[-3:])


def _format_catalog_context(items: list[dict]) -> str:
    """Format retrieved catalog items as a structured context block for the LLM."""
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


def _parse_response(text: str) -> dict:
    """
    Extract a JSON object from the LLM response text.
    Handles markdown code fences and stray surrounding text.
    Returns a valid response dict, or a fallback on parse failure.
    """
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
    Main agent entry point using Groq.

    Args:
        messages: Full conversation history as list of {"role": str, "content": str}.

    Returns:
        Dict matching the ChatResponse schema.
    """
    # Enforce max-turn limit
    user_turns = sum(1 for m in messages if m["role"] == "user")
    if user_turns > MAX_TURNS:
        return {
            "reply": "We've reached the maximum conversation length. Please start a new session.",
            "recommendations": [],
            "end_of_conversation": True,
        }

    # Retrieve relevant catalog items based on conversation
    query = _build_query(messages)
    catalog_items = retriever.search(query, top_k=20)

    # Build system prompt with rich catalog context
    catalog_context = _format_catalog_context(catalog_items)
    system_instruction = (
        SYSTEM_PROMPT
        + f"\n\nAVAILABLE ASSESSMENTS (use only these):\n\n{catalog_context}"
    )

    # Format history for Gemini (expects "user" or "assistant")
    formatted_msgs = [{"role": "user", "content": system_instruction}]
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        formatted_msgs.append({"role": role, "content": m["content"]})

    try:
        client = _get_client()
        response = client.generate_content(
            formatted_msgs,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=1024
            )
        )
        raw_text = response.text
    except Exception as e:
        return {
            **FALLBACK,
            "reply": f"Sorry, I encountered an error with the Gemini engine: {str(e)}. Please try again.",
        }

    return _parse_response(raw_text)
