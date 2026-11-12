import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging
import argparse

logger = logging.getLogger(__name__)

class RetrievalSystem:
    def __init__(self, chunks_file, faiss_index_file, model_path):
        self.chunks_df = None
        self.index = None
        self.model = None
        self.index_type = None

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
            index_description = str(type(self.index))
            if 'IndexFlatL2' in index_description or 'L2' in index_description:
                self.index_type = 'L2'
                logger.info(f"Loaded FAISS IndexFlatL2 with {self.index.ntotal} vectors")
            elif 'IndexFlatIP' in index_description or 'IP' in index_description:
                self.index_type = 'IP'
                logger.info(f"Loaded FAISS IndexFlatIP with {self.index.ntotal} vectors")
            else:
                self.index_type = 'L2'  # Default
                logger.warning(f"Unknown index type, assuming L2. Index: {index_description}")
        
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

    def _distance_to_similarity(self, distance):
        if self.index_type == 'IP':
            sim = distance  
        else:
            sim = 1 - (distance / 2)

        return max(0.0, min(1.0, float(sim)))

    def search(self, query, top_k=10):
        try:
            query_embedding = self.model.encode([query], normalize_embeddings=True)
            distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
            
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                chunk_info = self.chunks_df.iloc[idx]
                similarity = self._distance_to_similarity(dist)

                results.append({
                    'chunk_id': chunk_info.get('chunk_idx', f'chunk_{idx}'),
                    'chunk_text': chunk_info['chunk_text'],
                    'judul': chunk_info['judul'],
                    'author': chunk_info.get('first_author', 'unknown'),
                    'tahun': chunk_info.get('tahun_terbit', 'N/A'),
                    'url': chunk_info.get('url', '#'),
                    'section': chunk_info.get('chunk_section', 'unknown'),
                    'similarity': similarity,
                    'similarity_score': similarity
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


def _load_evaluation_queries(query_file):
    queries = pd.read_csv(query_file)
    queries = queries.rename(columns={
        'no': 'query_idx',
        'query_bahasa_indonesia': 'query',
        'reference_answer_bahasa_indonesia': 'reference_answer',
    })
    required = {'query', 'reference_answer'}
    missing = required.difference(queries.columns)
    if missing:
        raise ValueError(f'Missing query columns in {query_file}: {sorted(missing)}')
    if 'query_idx' not in queries.columns:
        queries.insert(0, 'query_idx', range(len(queries)))
    return queries


def export_semantic_candidates(
    query_file,
    faiss_index_file,
    chunks_file,
    model_path,
    output_file,
    top_k=10,
):
    """Retrieve and save one representative chunk per document for evaluation."""
    queries = _load_evaluation_queries(query_file)
    retriever = RetrievalSystem(chunks_file, faiss_index_file, model_path)
    rows = []

    for _, query_row in queries.iterrows():
        results = retriever.search(str(query_row['query']), top_k=top_k * 3)
        seen_titles = set()
        for result in results:
            title = str(result.get('judul', '')).strip()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            rows.append({
                'query_idx': int(query_row['query_idx']),
                'query': query_row['query'],
                'reference_answer': query_row['reference_answer'],
                'doc_title': title,
                'chunk_id': result.get('chunk_id', ''),
                'chunk_text': result.get('chunk_text', ''),
                'author': result.get('author', 'unknown'),
                'tahun': result.get('tahun', 'N/A'),
                'url': result.get('url', '#'),
                'section': result.get('section', 'unknown'),
                'retrieval_score': result.get('similarity_score', result.get('similarity', 0.0)),
                'rank': len(seen_titles),
                'retrieval_method': 'semantic',
            })
            if len(seen_titles) >= top_k:
                break

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_file, index=False)
    return result_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export semantic retrieval candidates')
    parser.add_argument('--queries', default='query_set_50.csv')
    parser.add_argument('--chunks', default='data/data_chunk.csv')
    parser.add_argument('--faiss-index', default='data/faiss_index.index')
    parser.add_argument('--model-path', default='models/sentence_transformer_model')
    parser.add_argument('--output', default='data/semantic_candidates.csv')
    parser.add_argument('--top-k', type=int, default=10)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = export_semantic_candidates(
        args.queries,
        args.faiss_index,
        args.chunks,
        args.model_path,
        output,
        args.top_k,
    )
    print(f'Saved {len(result)} semantic candidate rows to {output}')

