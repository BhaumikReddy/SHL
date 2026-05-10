"""
app/main.py

FastAPI application with:
  GET  /health  → {"status": "ok"}
  POST /chat    → ChatResponse (powered by Gemini + FAISS)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

from app.agent import get_agent_reply
from app.models import ChatRequest, ChatResponse, Recommendation

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent that recommends SHL assessments to hiring managers.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    result = get_agent_reply(messages)
    return ChatResponse(
        reply=result["reply"],
        recommendations=[
            Recommendation(**r) for r in result["recommendations"]
        ],
        end_of_conversation=result["end_of_conversation"],
    )
