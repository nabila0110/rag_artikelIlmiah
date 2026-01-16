import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class RetrievalSystem:
    def __init__(self, chunks_file, faiss_index_file, model_path):
        self.chunks_df = None
        self.index = None
        self.model = None

        self._ 