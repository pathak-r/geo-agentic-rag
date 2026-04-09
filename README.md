# Geo-Agentic RAG — Subsurface AI Assistant

An AI-powered subsurface engineering assistant that combines structured Volve production data with unstructured well PDFs using agentic RAG (LangChain tools + FAISS). **Web UI:** React (Vite) + Tailwind. **API:** FastAPI + Uvicorn (serves the built SPA and `/api/*` JSON).

Data source: Equinor [Volve Data Village](https://www.equinor.com/energy/volve-data-sharing) (Equinor Open Data Licence).

## Architecture

- **Frontend:** `frontend/` — React 18, TypeScript, Vite, Tailwind, Recharts.
- **Backend:** `backend/main.py` — FastAPI. Loads `src/` (data loader, anomalies, LangChain agent, FAISS search).
- **Legacy Streamlit UI** (optional): install `requirements-streamlit.txt` and run your own `streamlit run` entry if you still want it; the main maintained UI is the React app.

## Local development

**1. Python (repo root)**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # add OPENAI_API_KEY
```

Ensure `data/production/Volve production data.xlsx`, `data/pdfs/*.pdf`, and `data/faiss_index/` exist (run `python ingest.py` once if the index is missing).

**2. API + hot-reload UI (two terminals)**

Terminal A — FastAPI only (proxied by Vite):

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal B — React dev server:

```bash
cd frontend && npm install && npm run dev
```

Open **http://localhost:5173** (Vite proxies `/api` → port 8000).

**3. Production-style (single server)**

Build the SPA, then run Uvicorn from the repo root (it serves `frontend/dist` if present):

```bash
cd frontend && npm install && npm run build && cd ..
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open **http://127.0.0.1:8000/**.

## Deploy (generic PaaS: Railway, Render, Fly.io, Replit, etc.)

1. **Build:** Node 20+ → `cd frontend && npm ci && npm run build`.
2. **Python:** `pip install -r requirements.txt`.
3. **Start:** from repo root:

   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
   ```

4. **Env:** set `OPENAI_API_KEY` (and optionally `LLM_PROVIDER`, `ANTHROPIC_API_KEY` if you use Claude — add `langchain-anthropic` to dependencies).
5. **CORS:** if the UI is on another origin, set `CORS_ORIGINS` to a comma-separated list (default includes Vite dev ports).
6. **Data:** include `data/` in the deployment image or volume (same layout as locally).

## Streamlit Community Cloud

This stack is **not** a single-file Streamlit app anymore. To use Streamlit Cloud you would need a **custom Dockerfile** or a separate Streamlit-only branch. Prefer a container-friendly host with Node + Python build steps, or deploy the API and host the static build on any CDN.

## Project layout

```
backend/main.py       # FastAPI app + static mount
frontend/             # React SPA
src/                  # Shared Python: agent, tools, RAG, data_loader, anomaly
ingest.py             # PDF → FAISS (run locally / CI when PDFs change)
data/                 # Production xlsx, pdfs, faiss_index (large; often gitignored on public repos)
```

## Licence

Data: [Equinor Open Data Licence](https://www.equinor.com/energy/volve-data-sharing)  
Code: MIT
