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

def evaluate_retrieval(query_idx, retrieved_chunks, ground_truth_df, k=10, verbose=False):
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

    if verbose:
        print(f"\n[RETRIEVAL METRICS @k={k}]")
        print(f"Retrieved {len(retrieved_docs)} documents")
        print(f"Ground truth relevant docs: {len(relevant_docs)}")

    #relevant retrieved
    relevant_retrieved = sum(1 for doc in retrieved_docs if doc in relevant_docs)
    #highly relevant retrieved
    highly_relevant_retrieved = sum(1 for doc in retrieved_docs if doc in highly_relevant_docs)
    
    #calculate metrics\
    #PRECISION
    precision_k = relevant_retrieved / len(retrieved_docs) if len(retrieved_docs) > 0 else 0
    if verbose:
        print(f"\n[PRECISION@{k}] = {relevant_retrieved}/{len(retrieved_docs)} = {precision_k:.4f}")
    
    #RECALL
    recall_k = relevant_retrieved / len(relevant_docs) if len(relevant_docs) > 0 else 0
    if verbose:
        print(f"[RECALL@{k}] = {relevant_retrieved}/{len(relevant_docs)} = {recall_k:.4f}")
    
    #F1_SCORE_K 
    f1_k = 2 * precision_k * recall_k / (precision_k + recall_k) if (precision_k + recall_k) > 0 else 0
    if verbose:
        print(f"[F1-SCORE@{k}] = {f1_k:.4f}")
    
    # MRR (Mean Reciprocal Rank)
    mrr = 0
    mrr_pos = -1
    for i, doc in enumerate(retrieved_docs, 1):
        if doc in relevant_docs:
            mrr = 1 / i
            mrr_pos = i
            break
    if verbose:
        if mrr_pos > 0:
            print(f"\n[MRR@{k}] First relevant at position {mrr_pos} = 1/{mrr_pos} = {mrr:.4f}")
        else:
            print(f"[MRR@{k}] No relevant docs found = {mrr:.4f}")
    
    # MAP (Average Precision)
    precisions_at_relevan = []
    num_relevant_seen = 0
    if verbose:
        print(f"\n[AP@{k}] Calculation:")
    for i, doc in enumerate(retrieved_docs, 1):
        if doc in relevant_docs:
            num_relevant_seen += 1
            prec = num_relevant_seen/i
            precisions_at_relevan.append(prec)
            if verbose:
                print(f"  Position {i}: relevant found, P({num_relevant_seen}/{i}) = {prec:.4f}")

    # Legacy variant (hits-only denominator): mean over retrieved relevant positions only
    avg_precision_hits = np.mean(precisions_at_relevan) if precisions_at_relevan else 0

    # Standard AP denominator (sesuai rumus MAP klasik): total relevant docs
    ap_denom = len(relevant_docs)
    avg_precision = (sum(precisions_at_relevan) / ap_denom) if ap_denom > 0 else 0
    if verbose:
        if precisions_at_relevan:
            print(f"  AP@{k} (standard) = {' + '.join([f'{p:.4f}' for p in precisions_at_relevan])} / R={len(relevant_docs)}")
            print(f"                   = {sum(precisions_at_relevan):.4f} / {ap_denom} = {avg_precision:.4f}")
            print(f"  AP@{k} (hits-only) = {' + '.join([f'{p:.4f}' for p in precisions_at_relevan])} / {len(precisions_at_relevan)} = {avg_precision_hits:.4f}")
        else:
            print(f"  No relevant docs in top-{k}: AP@{k} = {avg_precision:.4f}")
    
    #nDCG_k
    relevances = [] #nanti outputnya kira*: relevances = [1, 0, 2, 2, 1] ini didapat dari relevance_score based on llm judge
    for doc in retrieved_docs:
        doc_gt = query_gt[query_gt['doc_title_norm'] == doc]
        rel = int(doc_gt['relevance_score'].iloc[0]) if len(doc_gt) > 0 else 0
        relevances.append(rel)
    
    def dcg_k(relevances, k):
        relevances = np.asarray(relevances)[:k]
        if relevances.size:
            gains = (2 ** relevances) - 1
            discounts = np.log2(np.arange(2, relevances.size + 2))
            return np.sum(gains / discounts)
        return 0.0
    
    dcg = dcg_k(relevances, len(retrieved_docs))
    if verbose:
        print(f"\n[nDCG@{k}] Relevance scores: {relevances}")
        print(f"  DCG calculation:")
        for i, rel in enumerate(relevances, 1):
            discount = np.log2(i + 1)
            gain = (2 ** rel) - 1
            contrib = gain / discount
            print(f"    Pos {i}: (2^{rel}-1)/log2({i+1}) = {gain}/{discount:.4f} = {contrib:.4f}")
        print(f"  DCG = {dcg:.4f}")
    
    # IDCG harus dihitung dari urutan ideal seluruh dokumen ground-truth untuk query ini
    # (bukan hanya dari dokumen yang berhasil diretrieve), lalu dipotong di k yang sama.
    ideal_relevances_all = sorted(query_gt['relevance_score'].astype(int).tolist(), reverse=True)
    idcg = dcg_k(ideal_relevances_all, len(retrieved_docs))
    ndcg = dcg / idcg if idcg > 0 else 0
    if verbose:
        print(f"  Ideal relevances (from all GT docs, top-{len(retrieved_docs)}): {ideal_relevances_all[:len(retrieved_docs)]}")
        print(f"  IDCG = {idcg:.4f}")
        print(f"  nDCG = {dcg:.4f} / {idcg:.4f} = {ndcg:.4f}")

    return {
        'Precision': precision_k,
        'Recall': recall_k,
        'F1 Score': f1_k,
        'MRR': mrr,
        'MAP': avg_precision,
        'MAP_HITS_ONLY': avg_precision_hits,
        'nDCG': ndcg,
        'Retrieved Docs': len(retrieved_docs),
        'Relevant Docs': len(relevant_docs),
        'Relevant Retrieved Docs': relevant_retrieved,
        'Highly Relevant Retrieved Docs': highly_relevant_retrieved,
        'Missed Relevant Docs': len(relevant_docs - set(retrieved_docs))
    }

#jalankan
def run_evaluation(query_df, ground_truth_df, retrieval_function, embedding_model, k_values=[5, 10], verbose=False):
    results = []

    # normalize ground truth titles once
    if 'doc_title_norm' not in ground_truth_df.columns:
        ground_truth_df = ground_truth_df.copy()
        ground_truth_df['doc_title_norm'] = ground_truth_df['doc_title'].apply(_normalize_title)

    for idx, (_, query_row) in enumerate(query_df.iterrows()): #(_) itu place holder, kan biasanya index yang disitu tapi ga perlu lagi karna index nya ga dipakek
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
                query_id, retrieved_chunks, ground_truth_df, k=k, 
                verbose=verbose  # Show detail for all queries if verbose
            )
            result.update({key: val for key, val in lab_metrics.items()
                      if not isinstance(val, list)})

            results.append(result)
        
    results_df = pd.DataFrame(results)

    # Print per-query summary if verbose
    if verbose:
        print("\n" + "="*80)
        print("PER-QUERY SUMMARY")
        print("="*80)
        for k in k_values:
            k_results = results_df[results_df['k']==k]
            print(f"\n@k={k}:")
            for idx, row in k_results.iterrows():
                print(f"  Query {row['query_idx']}: {row['query'][:60]}...")
                print(f"    Precision: {row['Precision']:.4f}, AP@{k}: {row['MAP']:.4f}, AP@{k}(hits-only): {row['MAP_HITS_ONLY']:.4f}, nDCG: {row['nDCG']:.4f}, MRR: {row['MRR']:.4f}")
    
    #aggregate all result
    for k in k_values:
        k_results = results_df[results_df['k']==k]

        print(f"\n{'='*80}")
        print(f"@{k} SUMMARY - Rata-Rata Hasil Evaluasi untuk {len(query_df)} queries:")
        print(f"{'='*80}")
        print(f"  Precision: {k_results['Precision'].mean():.4f} ± {k_results['Precision'].std():.4f}")
        # print(f"  Recall:    {k_results['Recall'].mean():.4f} ± {k_results['Recall'].std():.4f}")
        # print(f"  F1-Score:  {k_results['F1 Score'].mean():.4f} ± {k_results['F1 Score'].std():.4f}")
        print(f"  MAP@{k} (from AP@{k}): {k_results['MAP'].mean():.4f} ± {k_results['MAP'].std():.4f}")
        print(f"  MAP@{k} hits-only:     {k_results['MAP_HITS_ONLY'].mean():.4f} ± {k_results['MAP_HITS_ONLY'].std():.4f}")
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
    parser.add_argument("--verbose", action="store_true", help="Show detailed calculations for all queries")
    args = parser.parse_args()

    print("Loading queries and ground truth...")
    query_df = pd.read_csv(args.data_test)
    ground_truth_df = pd.read_csv(args.ground_truth)

    if RetrievalSystem is None:
        raise RuntimeError("RetrievalSystem import failed. Ensure utils/retrieval.py is available and dependencies installed.")

    print("Initializing retrieval system...")
    retriever = RetrievalSystem(args.chunks, args.faiss_index, args.model_path)

    print("Running evaluation...")
    run_evaluation(query_df, ground_truth_df, retriever.search, embedding_model=None, k_values=args.k, verbose=args.verbose)