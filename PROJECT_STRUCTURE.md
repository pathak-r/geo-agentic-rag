geo-agentic-rag/
│
├── .env.example          # API keys template
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
│
├── data/
│   ├── production/       # Put Volve production data.xlsx here
│   └── pdfs/             # Put drilling reports & completion reports here
│
├── src/
│   ├── __init__.py
│   ├── config.py         # App configuration & LLM provider setup
│   ├── data_loader.py    # Load and clean production data
│   ├── anomaly.py        # Anomaly detection on production data
│   ├── pdf_ingest.py     # PDF parsing, chunking, embedding
│   ├── vector_store.py   # FAISS index management
│   ├── tools.py          # Agent tools (production query, calculator)
│   ├── agent.py          # LangChain agent orchestration
│   └── llm.py            # LLM provider abstraction (swap OpenAI/Claude)
│
├── app.py                # Streamlit main app
└── ingest.py             # One-time script to build FAISS index from PDFs
