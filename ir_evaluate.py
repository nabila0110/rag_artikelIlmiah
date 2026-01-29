import pandas as pd
import numpy as np
from typing import Callable, List, Dict, Any
import re
import unicodedata

try:
    # Optional import for CLI usage; not required when imported as a module
    from utils.retrieval import RetrievalSystem
except Exception:
    RetrievalSystem = None


def _normalize_title(title: Any) -> str:
    """Robust title normalization for matching between chunks and ground truth.
    - Unicode NFKC normalize
    - Lowercase and strip
    - Replace fancy dashes/quotes
    - Collapse multiple spaces
    - Remove non-alphanumeric (keep spaces)
    """
    if title is None:
        return ""
    s = unicodedata.normalize('NFKC', str(title)).lower().strip()
    s = s.replace('–', '-').replace('—', '-').replace('’', "'").replace('“', '"').replace('”', '"')
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s

def evaluate_retrieval(query_idx, retrieved_chunks, ground_truth_df, k=10):
    #data_test dan ground_truth relevannya
    query_gt = ground_truth_df[ground_truth_df['query_idx'] == query_idx] #all AI-judge relevance rows for this one query.
    relevant_docs = set(query_gt[query_gt['relevance_score']>=1]['doc_title_norm'])
    highly_relevant_docs = set(query_gt[query_gt['relevance_score'] == 2]['doc_title_norm'])

    # get retrieved docs
    seen_docs = set()
    retrieved_docs = []

    for chunk in retrieved_chunks:
        if len(retrieved_docs) >= k:
            break
        doc_title = chunk['judul']
        doc_norm = _normalize_title(doc_title)
        if doc_norm not in seen_docs:
            retrieved_docs.append(doc_norm)
            seen_docs.add(doc_norm)

    #relevant retrieved
    relevant_retrieved = sum(1 for doc in retrieved_docs if doc in relevant_docs)
    #highly relevant retrieved
    highly_relevant_retrieved = sum(1 for doc in retrieved_docs if doc in highly_relevant_docs)
    
    #calculate metrics\
    #PRECISION
    precision_k = relevant_retrieved / len(retrieved_docs) if len(retrieved_docs) > 0 else 0
    #RECALL
    recall_k = relevant_retrieved / len(relevant_docs) if len(relevant_docs) > 0 else 0
    #F1_SCORE_K 
    f1_k = 2 * precision_k * recall_k / (precision_k + recall_k) if (precision_k + recall_k) > 0 else 0
    # MRR (Mean Reciprocal Rank)
    mrr = 0
    for i, doc in enumerate(retrieved_docs, 1):
        if doc in relevant_docs:
            mrr = 1 / i
            break
    #MAP (Mean Average Precision)
    precisions_at_relevan = []
    num_relevant_seen = 0
    for i, doc in enumerate(retrieved_docs, 1):
        if doc in relevant_docs:
            num_relevant_seen += 1
            precisions_at_relevan.append(num_relevant_seen/i)

    avg_precision = np.mean(precisions_at_relevan) if precisions_at_relevan else 0 #this mean if hte list exist or not 
    #nDCG_k
    relevances = [] #nanti outputnya kira*: relevances = [1, 0, 2, 2, 1] ini didapat dari relevance_score based on llm judge
    for doc in retrieved_docs:
        doc_gt = query_gt[query_gt['doc_title_norm'] == doc]
        rel = int(doc_gt['relevance_score'].iloc[0]) if len(doc_gt) > 0 else 0
        relevances.append(rel)
    def dcg_k(relevances, k):
        relevances = np.asarray(relevances)[:k]
        if relevances.size:
            return np.sum(relevances/np.log2(np.arange(2, relevances.size + 2)))
        return 0.0
    dcg = dcg_k(relevances, len(retrieved_docs))
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_k(ideal_relevances, len(retrieved_docs))
    ndcg = dcg / idcg if idcg > 0 else 0

    return {
        'Precision': precision_k,
        'Recall': recall_k,
        'F1 Score': f1_k,
        'MRR': mrr,
        'MAP': avg_precision,
        'nDCG': ndcg,
        'Retrieved Docs': len(retrieved_docs),
        'Relevant Docs': len(relevant_docs),
        'Relevant Retrieved Docs': relevant_retrieved,
        'Highly Relevant Retrieved Docs': highly_relevant_retrieved,
        'Missed Relevant Docs': len(relevant_docs - set(retrieved_docs))
    }

#jalankan
def run_evaluation(query_df, ground_truth_df, retrieval_function, embedding_model, k_values=[5, 10]):
    results = []

    # normalize ground truth titles once
    if 'doc_title_norm' not in ground_truth_df.columns:
        ground_truth_df = ground_truth_df.copy()
        ground_truth_df['doc_title_norm'] = ground_truth_df['doc_title'].apply(_normalize_title)

    for _, query_row in query_df.iterrows(): #(_) itu place holder, kan biasanya index yang disitu tapi ga perlu lagi karna index nya ga dipakek
        query_id = query_row.name #to get the index that i stupidly erase before p.s. i delete id btw cause i thought its nonsense
        query = query_row['query']

        #function retrieve - retrieve ALL chunks to cover all documents
        # Then dedupe and rank by document (corpus has ~11k chunks, 316 docs)
        rag_response = retrieval_function(query, top_k=15000)
    
        if isinstance(rag_response, dict):
            retrieved_chunks = rag_response['cited_references']
        else:
            retrieved_chunks = rag_response

        for k in k_values:
            result = {
                'query_idx': query_id,
                'query': query,
                'k': k
            }

            #retrieval_evaluate
            lab_metrics = evaluate_retrieval(
                query_id, retrieved_chunks, ground_truth_df, k=k
            )
            result.update({key: val for key, val in lab_metrics.items()
                      if not isinstance(val, list)})

            results.append(result)
        
    results_df = pd.DataFrame(results)

    #aggregate all result
    for k in k_values:
        k_results = results_df[results_df['k']==k]

        print(f"\n@{k} Rata-Rata Hasil Evaluasi untuk {len(query_df)} queries:")
        print(f"  Precision: {k_results['Precision'].mean():.4f} ± {k_results['Precision'].std():.4f}")
        print(f"  Recall:    {k_results['Recall'].mean():.4f} ± {k_results['Recall'].std():.4f}")
        print(f"  F1-Score:  {k_results['F1 Score'].mean():.4f} ± {k_results['F1 Score'].std():.4f}")
        print(f"  MAP:       {k_results['MAP'].mean():.4f} ± {k_results['MAP'].std():.4f}")
        print(f"  nDCG:      {k_results['nDCG'].mean():.4f} ± {k_results['nDCG'].std():.4f}")
        print(f"  MRR:       {k_results['MRR'].mean():.4f} ± {k_results['MRR'].std():.4f}")
    results_df.to_csv('eval_results.csv', index=False)

    return results_df


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval metrics")
    parser.add_argument("--k", type=int, nargs="*", default=[5, 10], help="List of k values for @k metrics")
    parser.add_argument("--data_test", type=str, default=os.path.join("data", "data_test.csv"), help="Path to queries CSV")
    parser.add_argument("--ground_truth", type=str, default=os.path.join("data", "ground_truth_data.csv"), help="Path to ground truth CSV")
    parser.add_argument("--chunks", type=str, default=os.path.join("data", "data_chunk.csv"), help="Path to chunks CSV")
    parser.add_argument("--faiss_index", type=str, default=os.path.join("data", "faiss_index.index"), help="Path to FAISS index file")
    parser.add_argument("--model_path", type=str, default=os.path.join("models", "sentence_transformer_model"), help="Path to sentence transformer model dir")
    args = parser.parse_args()

    print("Loading queries and ground truth...")
    query_df = pd.read_csv(args.data_test)
    ground_truth_df = pd.read_csv(args.ground_truth)

    if RetrievalSystem is None:
        raise RuntimeError("RetrievalSystem import failed. Ensure utils/retrieval.py is available and dependencies installed.")

    print("Initializing retrieval system...")
    retriever = RetrievalSystem(args.chunks, args.faiss_index, args.model_path)

    print("Running evaluation...")
    run_evaluation(query_df, ground_truth_df, retriever.search, embedding_model=None, k_values=args.k)