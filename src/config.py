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
CHUNK_SIZE = 1000       # Target size for SemanticChunker (guidance only)
CHUNK_OVERLAP = 200
MAX_CHUNK_SIZE = 1500   # Hard ceiling: any chunk larger than this is re-split
TOP_K_RESULTS = 10

# --- Anomaly Detection ---
ANOMALY_WINDOW = 30  # Rolling window in days
ANOMALY_THRESHOLD = 2.0  # Standard deviations for anomaly flagging
