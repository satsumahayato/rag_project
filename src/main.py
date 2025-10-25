# A lightwight, custom RAG pipeline is implemented manually, giving full transparency and flexibility.
# The orchestration sequence is explicit:
# 1. Document ingestion -> parse_documents.py
# 2. Chunking -> chunk_text.py
# 3. Embedding -> embed_corpus.py (using OpenAI embeddings API directly)
# 4. Storage and retrieval -> ChromaDB client API
# 5. Answer generation -> direct call to OpenAI chat.completions endpoint in main.py

from pathlib import Path
from fastapi.responses import JSONResponse

# src/main.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timezone
import time

from src.retrieval.retrieve_chunks import retrieve_top_k
from openai import OpenAI
import os, time

from typing import Union


app = FastAPI(title="RAG Policy Assistant", version="0.1.0")

# --- CORS (relaxed for local dev; tighten for prod) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change to your deployed origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data models ---
class Citation(BaseModel):
    title: Optional[str] = None
    doc_id: Optional[str] = None
    snippet: str
    page: Optional[Union[int, str]] = None
    source_url: Optional[str] = None

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question about company policies")

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    latency_ms: float

# --- Routes ---
@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "service": app.title,
        "version": app.version,
        "time_utc": datetime.now(timezone.utc).isoformat()
    })

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    t0 = time.perf_counter()
    try:
        # 1) Retrieve
        chunks = retrieve_top_k(
            query=req.question,
            k=4,
            db_path=Path(".chromadb"),
            model_backend="openai",
            openai_model="text-embedding-3-small"
        )

        if not chunks:
            return ChatResponse(
                answer="I couldn't find relevant information in the corpus.",
                citations=[],
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        # 2) Build context
        context_texts = [f"[{c.get('title','Untitled')} p{c.get('page_start','?')}] {c['text']}" for c in chunks]
        context = "\n\n".join(context_texts)

        # 3) Prompt
        prompt = f"""You are a helpful assistant that answers questions based strictly on the provided context.
If the answer is not in the context, say you cannot find it.
Always cite sources by document title and page number.

CONTEXT:
{context}

QUESTION:
{req.question}

ANSWER:"""

        # 4) Generate
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer = completion.choices[0].message.content.strip()

        # 5) Citations
        citations = [
        {
            "title": c.get("title") or "Source Document",
            "page": c.get("page_start"),
            "snippet": (c["text"][:150] + "...") if len(c["text"]) > 150 else c["text"],
        }
        for c in chunks
        ]
        
        return ChatResponse(
            answer=answer,
            citations=citations,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    except Exception as e:
        # Return structured JSON error so the front-end doesn't choke on HTML
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "detail": str(e)},
        )

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    # Ultra-minimal UI so you can test /chat without building a front-end yet.
    html = """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width,initial-scale=1"/>
      <title>RAG Policy Assistant</title>
      <style>
        body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
               margin: 2rem; line-height: 1.4; }
        .wrap { max-width: 800px; margin: auto; }
        textarea { width: 100%; height: 120px; padding: .75rem; font-size: 1rem; }
        button { padding: .6rem 1rem; font-size: 1rem; cursor: pointer; }
        .answer { white-space: pre-wrap; background: #fafafa; padding: 1rem; border: 1px solid #eee; border-radius: 8px; }
        .meta { color: #666; font-size: .9rem; }
        .cit { margin-top: .25rem; padding-left: 1rem; border-left: 3px solid #ddd; }
      </style>
    </head>
    <body>
      <div class="wrap">
        <h1>RAG Policy Assistant</h1>
        <p>Ask a question about company policies. This is a minimal demo; retrieval and citations will be wired up next.</p>
        <textarea id="q" placeholder="e.g., How does PTO accrue for new hires?"></textarea><br/>
        <button id="ask">Ask</button>
        <p class="meta" id="status"></p>
        <div class="answer" id="ans" style="display:none;"></div>
        <div id="cits"></div>
      </div>
      <script>
  const askBtn = document.getElementById('ask');
  const qEl = document.getElementById('q');
  const ansEl = document.getElementById('ans');
  const citsEl = document.getElementById('cits');
  const statusEl = document.getElementById('status');

  askBtn.onclick = async () => {
    const question = qEl.value.trim();
    if (!question) { alert("Please enter a question."); return; }
    ansEl.style.display = 'none';
    citsEl.innerHTML = '';
    statusEl.textContent = 'Thinking...';
    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      });

      let data;
      if (res.ok) {
        data = await res.json();
      } else {
        // try json error, else text
        try { data = await res.json(); }
        catch { throw new Error(await res.text()); }
        throw new Error(data?.detail || data?.error || 'Server error');
      }

      ansEl.textContent = data.answer || '(no answer)';
      ansEl.style.display = 'block';
      statusEl.textContent = `Latency: ${typeof data.latency_ms === 'number' ? data.latency_ms : 'n/a'} ms`;

      if (Array.isArray(data.citations)) {
        const details = document.createElement('details');
        details.innerHTML = "<summary>Sources</summary>";
        data.citations.forEach(c => {
          const div = document.createElement('div');
          div.className = 'cit';
          const label = c.title || c.doc_id || 'Source';
          const page = c.page ? ` (p. ${c.page})` : '';
          div.innerHTML = `<strong>${label}${page}:</strong> ${c.snippet || ''}`;
          details.appendChild(div);
        });
        citsEl.appendChild(details);
      }
    } catch (err) {
      statusEl.textContent = 'Error: ' + (err?.message || err);
      ansEl.textContent = '(no answer)';
      ansEl.style.display = 'block';
    }
  };
</script>
    </body>
    </html>
    """
    return HTMLResponse(html)