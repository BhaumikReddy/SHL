"""
app/main.py

FastAPI application with:
  GET  /health  → {"status": "ok"}
  POST /chat    → ChatResponse (stub for now, wired in Stage 4)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    # Stub — will be replaced with real agent logic in Stage 4
    return ChatResponse(
        reply="stub",
        recommendations=[],
        end_of_conversation=False,
    )
