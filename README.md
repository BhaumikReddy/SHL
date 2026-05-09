# SHL Assessment Recommender

A conversational API that helps hiring managers find the right SHL assessments for their roles.

## What it does

- Takes a conversation history and returns a natural-language reply plus a list of SHL assessment recommendations
- Uses semantic search (FAISS + sentence-transformers) over the scraped SHL catalog to surface relevant assessments
- Powered by Google Gemini for natural language understanding and generation
- Stateless REST API — no session storage; full conversation history is sent with every request

## Quickstart (local)

```bash
# 1. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Gemini API key
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 4. (Already done) Scrape the SHL catalog
python scripts/scrape.py

# 5. Build the FAISS vector index
python scripts/build_index.py

# 6. Run the API
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

## API

### `GET /health`
Returns `{"status": "ok"}`.

### `POST /chat`
**Request body:**
```json
{
  "messages": [
    {"role": "user", "content": "I need an assessment for a software engineer role"}
  ]
}
```

**Response:**
```json
{
  "reply": "string",
  "recommendations": [
    {"name": "string", "url": "string", "test_type": "string"}
  ],
  "end_of_conversation": false
}
```

- `recommendations` is `[]` when the agent is still clarifying.
- `end_of_conversation` is `true` when the user appears satisfied.

## Environment variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key (get one free at [aistudio.google.com](https://aistudio.google.com/app/apikey)) |

## Project structure

```
app/          FastAPI app, agent logic, retriever, models
scripts/      Scraper and index builder
data/         Scraped catalog (catalog.json) and FAISS index
```
