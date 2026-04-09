# 🛢️ Geo-Agentic RAG — Subsurface AI Assistant

An AI-powered subsurface engineering assistant that combines structured production data with unstructured well documentation using agentic RAG (Retrieval Augmented Generation). Built on real data from Equinor's [Volve field](https://www.equinor.com/energy/volve-data-sharing) — the most comprehensive open subsurface dataset from the Norwegian Continental Shelf.

## What It Does

A reservoir or production engineer can ask natural language questions like:

- *"What's causing the water cut increase in well F-11?"*
- *"Based on the 1997 drilling reports, what formations were encountered?"*
- *"Calculate the decline rate for F-1 C over the last 12 months"*
- *"What anomalies have been detected across all wells?"*

The AI agent decides whether to query production data, search well documents, or do both — then synthesizes a coherent answer with source citations.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                 Streamlit UI                     │
│  ┌──────────────┐ ┌──────────┐ ┌──────────────┐ │
│  │  Production   │ │   Chat   │ │   Anomaly    │ │
│  │  Dashboard    │ │Interface │ │  Detection   │ │
│  └──────────────┘ └────┬─────┘ └──────────────┘ │
│                        │                         │
│  ┌─────────────────────▼───────────────────────┐ │
│  │           LangChain Agent                    │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │ │
│  │  │Production│ │ Document │ │  Calculator  │ │ │
│  │  │  Query   │ │  Search  │ │    Tools     │ │ │
│  │  └────┬─────┘ └────┬─────┘ └──────┬───────┘ │ │
│  └───────┼────────────┼──────────────┼─────────┘ │
│          │            │              │           │
│  ┌───────▼──────┐ ┌───▼────────┐ ┌───▼────────┐ │
│  │   Pandas     │ │   FAISS    │ │  Anomaly   │ │
│  │  DataFrame   │ │   Vector   │ │  Detection │ │
│  │              │ │   Store    │ │  Engine    │ │
│  └───────┬──────┘ └─────┬──────┘ └────────────┘ │
│          │              │                        │
│  ┌───────▼──────┐ ┌─────▼──────┐                 │
│  │  Volve       │ │  Well PDFs │                 │
│  │  Production  │ │  (Reports) │                 │
│  │  Data (.xlsx)│ │            │                 │
│  └──────────────┘ └────────────┘                 │
└─────────────────────────────────────────────────┘
```

## Tech Stack

- **LLM**: OpenAI GPT-4o (swappable to Claude via config)
- **Agent Framework**: LangChain
- **Vector Store**: FAISS
- **Embeddings**: OpenAI text-embedding-3-small
- **Data Processing**: pandas, numpy
- **Anomaly Detection**: Statistical (rolling z-score)
- **Frontend**: Streamlit
- **Charts**: Plotly

## Data Source

[Equinor Volve Data Village](https://www.equinor.com/energy/volve-data-sharing) — released under Equinor Open Data Licence for research and development.

- **Production Data**: Daily & monthly production for 7 wellbores (2008-2016)
- **Well Documents**: Daily drilling reports, completion reports, final well reports

## Setup

```bash
# Clone
git clone https://github.com/pathak-r/geo-agentic-rag.git
cd geo-agentic-rag

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your OpenAI API key

# Place data files (commit these paths for Streamlit Cloud — see below)
# - data/production/Volve production data.xlsx
# - data/pdfs/*.pdf (drilling reports, completion reports)
# - data/faiss_index/ (after ingest, or build on your machine and push)

# Build the vector index (one-time, if not already in repo)
python ingest.py

# Run the app
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub, including **`data/production/`**, **`data/pdfs/`**, and **`data/faiss_index/`** (run `python ingest.py` locally first if the index is not in git yet). Respect GitHub file-size limits (use [Git LFS](https://git-lfs.com/) if individual files exceed ~50–100&nbsp;MB).
2. In [Streamlit Community Cloud](https://streamlit.io/cloud), create an app: repository **pathak-r/geo-agentic-rag**, branch **`main`**, main file **`app.py`**.
3. Under **Advanced settings → Secrets**, paste (replace with your real key):

```toml
OPENAI_API_KEY = "sk-..."
```

Optional (defaults to OpenAI if omitted):

```toml
LLM_PROVIDER = "openai"
```

If you use **Anthropic** instead, set `LLM_PROVIDER = "anthropic"`, add `ANTHROPIC_API_KEY = "..."`, and ensure **`langchain-anthropic`** is installed (add it to `requirements.txt` or install via Cloud dependencies).

Secrets are copied into the process environment before the app loads `src.config`, so the LangChain/OpenAI clients pick them up the same as a local `.env`.

## Project Context

This project demonstrates how agentic AI can accelerate subsurface engineering workflows — similar in concept to [AIQ's ENERGYai](https://aiqintelligence.ai/) platform, which uses LLMs and agentic AI to automate upstream oil & gas operations across ADNOC's fields.

The same architectural pattern (multi-source RAG + domain-specific tools + agentic reasoning) applies to any industrial domain where engineers need to synthesize insights across structured operational data and unstructured technical documentation.

## License

Data: [Equinor Open Data Licence](https://www.equinor.com/energy/volve-data-sharing)
Code: MIT
