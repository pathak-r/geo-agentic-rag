geo-agentic-rag/
├── .env.example
├── requirements.txt           # FastAPI + RAG
├── requirements-streamlit.txt # optional Streamlit + Plotly
├── backend/
│   └── main.py                # FastAPI: /api/* + frontend/dist
├── frontend/                  # React + Vite + Tailwind
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── anomaly.py
│   ├── pdf_ingest.py
│   ├── vector_store.py
│   ├── tools.py
│   ├── agent.py
│   └── llm.py
├── data/
│   ├── production/
│   ├── pdfs/
│   └── faiss_index/
├── ingest.py
└── README.md
