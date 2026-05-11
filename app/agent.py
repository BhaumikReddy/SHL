import json
import os
import re

from dotenv import load_dotenv
from groq import Groq  # Switched from google.genai

from . import retriever

load_dotenv()

_client = None

def _get_client() -> Groq:
    global _client
    if _client is None:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "GROQ_API_KEY environment variable is missing. "
                "Please set it in your .env file."
            )
        _client = Groq(api_key=key)
    return _client

_MODEL = "llama-3.3-70b-versatile"

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
    * job_levels, duration, languages, remote, adaptive, keys.
- Refuse off-topic questions.
- Response format — ALWAYS respond with ONLY valid JSON.

JSON Structure:
{
  "reply": "explanation",
  "recommendations": [{"name": "...", "url": "...", "test_type": "..."}],
  "end_of_conversation": false
}
"""

FALLBACK = {"reply": "", "recommendations": [], "end_of_conversation": False}

def _build_query(messages: list[dict]) -> str:
    user_texts = [m["content"] for m in messages if m["role"] == "user"]
    return " ".join(user_texts[-3:])

def _format_catalog_context(items: list[dict]) -> str:
    lines = []
    for item in items:
        test_type = get_test_type(item)
        lines.append(
            f"- Name: {item['name']}\n"
            f"  URL: {item['link']}\n"
            f"  Test Type Code: {test_type}\n"
            f"  Description: {item.get('description', '')[:150]}"
        )
    return "\n\n".join(lines)

def _parse_response(text: str) -> dict:
    # Remove potential markdown code blocks
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    
    try:
        data = json.loads(text)
        return {
            "reply": data.get("reply", ""),
            "recommendations": data.get("recommendations", [])[:10],
            "end_of_conversation": bool(data.get("end_of_conversation", False))
        }
    except Exception:
        # Fallback search for JSON within the string if the model added conversational filler
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return data
            except: pass
        return {**FALLBACK, "reply": "I apologize, I had trouble formatting the response. Could you try again?"}

def get_agent_reply(messages: list[dict]) -> dict:
    user_turns = sum(1 for m in messages if m["role"] == "user")
    if user_turns > MAX_TURNS:
        return {
            "reply": "Session limit reached. Please start a new chat.",
            "recommendations": [],
            "end_of_conversation": True,
        }

    query = _build_query(messages)
    catalog_items = retriever.search(query, top_k=10)
    catalog_context = _format_catalog_context(catalog_items)

    # Groq uses a standard OpenAI-style messages list
    chat_messages = [
        {"role": "system", "content": SYSTEM_PROMPT + f"\n\nAVAILABLE ASSESSMENTS:\n{catalog_context}"}
    ]
    
    # Append conversation history
    for m in messages:
        chat_messages.append({"role": m["role"], "content": m["content"]})

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=_MODEL,
            messages=chat_messages,
            response_format={"type": "json_object"}, # Forces JSON mode
            temperature=0.2 # Lower temperature for better catalog matching
        )
        raw_text = response.choices[0].message.content
        return _parse_response(raw_text)
        
    except Exception as e:
        return {
            **FALLBACK,
            "reply": f"Groq Error: {str(e)}",
        }