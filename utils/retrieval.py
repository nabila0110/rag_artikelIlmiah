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

        self._load_data(chunks_file)
        self._load_index(faiss_index_file)
        self._load_model(model_path)

        logger.info("Retrieval system initialized successfully")
    
    def _load_data(self, chunks_file):
        try:
            self.chunks_df = pd.read_csv(chunks_file)
            logger.info(f"loaded{len(self.chunks_df)} chunks from {chunks_file}")
        except Exception as e:
            logger.error(f"Error loading chunks: {e}")
            raise

    def _load_index(self, faiss_index_file):
        try:
            self.index = faiss.read_index(str(faiss_index_file))
            logger.info(f"loaded faiss index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            raise

    def _load_model(self, model_path):
        try:
            self.model = SentenceTransformer(str(model_path))
            logger.info(f"loaded embedding model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    def search(self, query, top_k=30):
        try:
            query_embedding = self.model.encode([query], normalize_embeddings=True)
            distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
            
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                chunk_info = self.chunks_df.iloc[idx]

                results.append({
                    'chunk_id': chunk_info.get('chunk_idx', f'chunk_{idx}'),
                    'chunk_text': chunk_info['chunk_text'],
                    'judul': chunk_info['judul'],
                    'author': chunk_info.get('first_author', 'unknown'),
                    'tahun': chunk_info.get('tahun_terbit', 'N/A'),
                    'url': chunk_info.get('url', '#'),
                    'section': chunk_info.get('chunk_section', 'unknown'),
                    'similarity_score': 1 - (dist ** 2) / 2 
                })

            logger.info(f"Retrieved {len(results)} results for query: '{query}'")
            return results

        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise
            
    def get_statistics(self):
        return {
            'total_chunks': len(self.chunks_df),
            'total_documents': self.chunks_df['judul'].nunique(),
            'index_size': self.index.ntotal,
            'embedding_dimension': self.index.d
        }   

