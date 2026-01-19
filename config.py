import os
from pathlib import Path

class Config:
    #flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = False
    TESTING = False

    #paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    MODELS_DIR = BASE_DIR / 'models'

    #data files
    CHUNKS_FILE = DATA_DIR / 'data_chunk.csv'
    FAISS_INDEX_FILE = DATA_DIR / 'faiss_index.index'
    EMBEDDINGS_FILE = DATA_DIR / 'embeddings.npy'

    #model settings
    EMBEDDING_MODEL_NAME = 'infloat/multilingual-e5-base'
    EMBEDDING_MODEL_PATH = MODELS_DIR / 'sentence_transformer_model'

    #llm settings
    LLM_MODEL = 'gemma2:9b'
    LLM_TEMPERATURE = 0.3
    LLM_MAX_TOKENS = 500

    #retrieval settings
    DEFAULT_TOP_K = 20
    MAX_TOP_K = 30

    #generation settings
    MAX_CONTEXT_CHUNKS = 5

    #api settings
    MAX_REQUESTS_PER_MINUTE = 60
    REQUEST_TIMEOUT = 30  # seconds

class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = True

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    DEBUG = True
    TESTING = True

#config dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}