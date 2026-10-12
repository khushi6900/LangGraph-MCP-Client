from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent   # Mcp_client_chatbot_with_Rag/
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_DB_PATH = DATA_DIR / "vector_store.faiss"
DB_PATH = DATA_DIR / "chatbot.db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)