import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Configuration ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "anthropic"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")
OPENAI_MODEL = "gpt-4o"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
EMBEDDING_MODEL = "text-embedding-3-small"

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PRODUCTION_DATA_PATH = os.path.join(DATA_DIR, "production", "Volve production data.xlsx")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")

# --- RAG Configuration ---
CHUNK_SIZE = 1000       # SemanticChunker guidance or fixed splitter chunk size
CHUNK_OVERLAP = 200
MAX_CHUNK_SIZE = 1500   # Hard ceiling when using semantic chunking + secondary split
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "8"))

# semantic = FAISS only (default). hybrid = BM25 + FAISS + RRF
RAG_MODE = os.getenv("RAG_MODE", "semantic").strip().lower()
RAG_MULTI_QUERY = os.getenv("RAG_MULTI_QUERY", "0").strip().lower() in ("1", "true", "yes")
RAG_MULTI_QUERY_N = int(os.getenv("RAG_MULTI_QUERY_N", "3"))
RAG_RERANK = os.getenv("RAG_RERANK", "1").strip().lower() in ("1", "true", "yes")
RAG_RERANK_POOL = int(os.getenv("RAG_RERANK_POOL", "40"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
# pdf ingest: semantic | fixed
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "semantic").strip().lower()
DOC_SEARCH_EXCERPT_CHARS = int(os.getenv("DOC_SEARCH_EXCERPT_CHARS", "2400"))

# --- Anomaly Detection ---
ANOMALY_WINDOW = 30
ANOMALY_THRESHOLD = 2.0
