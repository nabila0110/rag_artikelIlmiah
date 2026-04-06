from pathlib import Path

class Config:
    # Flask settings
    SECRET_KEY = 'dev-secret-key'
    DEBUG = True
    TESTING = False

    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    MODELS_DIR = BASE_DIR / 'models'

    # Data files
    CHUNKS_FILE = DATA_DIR / 'data_chunk.csv'
    FAISS_INDEX_FILE = DATA_DIR / 'faiss_index.index'
    EMBEDDINGS_FILE = DATA_DIR / 'embeddings.npy'

    # Model settings
    EMBEDDING_MODEL_NAME = 'infloat/multilingual-e5-base'
    EMBEDDING_MODEL_PATH = MODELS_DIR / 'sentence_transformer_model'

    # LLM settings
    LLM_MODEL = 'gemma2:9b'
    LLM_TEMPERATURE = 0.3
    LLM_MAX_TOKENS = 500

    # Retrieval settings
    DEFAULT_TOP_K = 10
    MAX_TOP_K = 100

    # Generation settings
    MAX_CONTEXT_CHUNKS = 5

    # API settings
    MAX_REQUESTS_PER_MINUTE = 60
    REQUEST_TIMEOUT = 30  # seconds