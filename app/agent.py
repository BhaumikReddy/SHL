"""
app/agent.py

Agent logic: builds a query from conversation history, retrieves relevant
catalog items via FAISS, calls Groq (llama-3.3-70b-versatile), and parses
the JSON response.
"""

import json
import os
import re

from dotenv import load_dotenv
from groq import Groq

from . import retriever

load_dotenv()

_client = None

def _get_client():
    global _client
    if _client is None:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY environment variable is missing. Please set it in your .env file.")
        _client = Groq(api_key=key)
    return _client

_MODEL = "llama-3.3-70b-versatile"

MAX_TURNS = 8  # hard cap on conversation length

SYSTEM_PROMPT = """\
You are an SHL assessment recommender. Your job is to help hiring managers find \
the right SHL assessments from the official catalog.

Rules:
- Only recommend assessments from the catalog context provided to you.
- Never invent assessment names or URLs.
- If the user's request is too vague, ask ONE clarifying question before recommending.
- Once you have enough context (role, what they want to measure), recommend 1-10 assessments.
- If the user asks to compare assessments, compare only using the catalog data given.
- Refuse off-topic questions (legal, general HR advice, anything unrelated to SHL assessments).
- Do not recommend on the very first turn if the query is vague.
- When you are ready to recommend, respond ONLY with valid JSON matching this schema:
  {"reply": "...", "recommendations": [{"name": "...", "url": "...", "test_type": "..."}], "end_of_conversation": false}
- When still clarifying, respond ONLY with valid JSON:
  {"reply": "your question here", "recommendations": [], "end_of_conversation": false}
- Set end_of_conversation to true only when the user seems satisfied with the shortlist.
- Always return valid JSON. Never return plain text outside of JSON.
"""

FALLBACK = {"reply": "", "recommendations": [], "end_of_conversation": False}


def _build_query(messages: list[dict]) -> str:
    """Extract a search query from the conversation — combine recent user turns."""
    user_texts = [m["content"] for m in messages if m["role"] == "user"]
    return " ".join(user_texts[-3:])


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
    catalog_items = retriever.search(query, top_k=10)

    # Build system prompt with context
    catalog_block = json.dumps(catalog_items, indent=2, ensure_ascii=False)
    system_instruction = (
        SYSTEM_PROMPT
        + f"\n\nAVAILABLE ASSESSMENTS (use only these):\n{catalog_block}"
    )

    # Format history for Groq (expects "user" or "assistant")
    formatted_msgs = [{"role": "system", "content": system_instruction}]
    for m in messages:
        role = "assistant" if m["role"] == "assistant" else "user"
        formatted_msgs.append({"role": role, "content": m["content"]})

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=_MODEL,
            messages=formatted_msgs,
            temperature=0.2,
            max_tokens=1024,
            response_format={"type": "json_object"}  # enforce strict JSON format
        )
        raw_text = response.choices[0].message.content
    except Exception as e:
        return {
            **FALLBACK,
            "reply": f"Sorry, I encountered an error with the Groq engine: {str(e)}. Please try again.",
        }

    return _parse_response(raw_text)
