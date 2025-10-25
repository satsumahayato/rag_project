RAG Policy Assistant
A Retrieval-Augmented Generation (RAG) web app built with FastAPI, ChromaDB, and OpenAI. It allows users to ask natural-language questions about internal company policies (PDF, Markdown, or text files) and receive grounded, cited answers.
Features
• Parses, cleans, and normalizes mixed-format documents (PDF, HTML, Markdown, TXT)
• Splits text into semantically meaningful chunks
• Generates embeddings (OpenAI or Hugging Face) and stores them in a vector database
• Retrieves the most relevant chunks for a user query
• Uses GPT-based generation to produce contextual, cited answers
• Includes a simple, interactive FastAPI web interface
Project Structure
ai-engineering-group-project/
│
├── src/
│   ├── main.py                        # FastAPI app entry point
│   ├── ingestion/
│   │   ├── parse_documents.py         # Parses & cleans PDFs, HTML, MD, TXT
│   │   └── validate_and_preview.py    # Validates corpus & adds metadata
│   ├── embeddings/
│   │   ├── chunk_text.py              # Semantic chunking
│   │   └── embed_corpus.py            # Embedding & vector storage
│   ├── retrieval/
│   │   └── retrieve_chunks.py         # Top-k similarity search
│   └── ...
│
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Ignore local data & venv
└── README.md                          # Project overview
Local Setup
1. Clone the repository:
   git clone https://github.com/satsumahayato/rag_project.git
   cd rag_project

2. Create a virtual environment:
   python3 -m venv venv
   source venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

4. Set your environment key:
   export OPENAI_API_KEY='sk-...'

Build the Knowledge Base
1. Place your source documents (PDFs, .md, .txt, etc.) in: data/raw_docs/
2. Run ingestion and chunking:
   python -m src.ingestion.parse_documents --in data/raw_docs --out data/processed/corpus_clean.jsonl
   python -m src.embeddings.chunk_text --in data/processed/corpus_clean.jsonl --out data/processed/corpus_chunks.jsonl
3. Generate embeddings and store them in Chroma:
   python -m src.embeddings.embed_corpus --in data/processed/corpus_chunks.jsonl --db .chromadb --model openai
Run the App
uvicorn src.main:app --reload

Then open your browser to http://127.0.0.1:8000
Deployment (Render)
1. Push this repo to GitHub (public or private).
2. Create a Web Service at https://render.com.
3. Configure:
   - Build Command: pip install -r requirements.txt
   - Start Command: uvicorn src.main:app --host 0.0.0.0 --port 10000
4. Add environment variable OPENAI_API_KEY.
5. Wait for deployment → visit your live app URL.
Tech Stack
• Python 3.10+
• FastAPI – web framework
• ChromaDB – vector database for embeddings
• OpenAI / Hugging Face – text embedding models
• Uvicorn – ASGI web server
Example Prompt
User:
   'What is the PTO accrual rate for full-time employees?'

Response:
   'Full-time employees accrue paid time off at 1.25 days per month according to the Employee Handbook (p. 12).'
Credits
Developed as part of the AI Engineering Group Project (Quantic MSSE) by Hidefusa Okabe (2025).
<img width="432" height="647" alt="image" src="https://github.com/user-attachments/assets/9854bbfe-0de2-4b48-ab67-47b36126931b" />
