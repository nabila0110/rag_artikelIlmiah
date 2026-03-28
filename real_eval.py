# eval_machine_learning.py
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import spacy
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
import unicodedata
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from utils.retrieval import RetrievalSystem
    from utils.generation import GenerationSystem
except Exception as e:
    print(f"Warning: Could not import utils: {e}")
    RetrievalSystem = None
    GenerationSystem = None


def _normalize_title(title):
    """Normalize document title for matching"""
    if title is None:
        return ""
    s = unicodedata.normalize('NFKC', str(title)).lower().strip()
    s = s.replace('–', '-').replace('—', '-').replace("'", "'").replace('"', '"').replace('"', '"')
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s


def load_spacy_model(name="xx_ent_wiki_sm"):
    """Load spaCy model with fallback"""
    fallback_models = [name, "xx_ent_wiki_sm", "xx_sent_ud_sm"]
    
    for model_name in fallback_models:
        try:
            print(f"Loading spaCy model: {model_name}...", end=" ")
            nlp = spacy.load(model_name)
            print("✓ Loaded")
            if model_name != name:
                print(f"  ℹ Using fallback model: {model_name}")
            return nlp
        except OSError:
            print(f" ✗ Not found")
            try:
                print(f"  Attempting to download {model_name}...")
                from spacy.cli import download
                download(model_name)
                nlp = spacy.load(model_name)
                print(f"  ✓ Downloaded and loaded")
                if model_name != name:
                    print(f"  ℹ Using fallback model: {model_name}")
                return nlp
            except Exception:
                if model_name == fallback_models[-1]:
                    raise RuntimeError(f"Failed to load any spaCy model. Tried: {', '.join(fallback_models)}")
                continue


def eval_machine_learning_detailed():
    """
    Evaluate query pertama dari data_test.csv dengan detail step-by-step
    - Retrieval evaluation menggunakan ground_truth_data.csv
    - Generation evaluation menggunakan reference_answer dari query_test.csv
    """
    print("\n" + "="*80)
    print("EVALUASI DETAIL: Query Pertama dari data_test.csv")
    print("="*80 + "\n")
    
    # Load data
    print("[1] Loading data...")
    data_test = pd.read_csv("data/data_test.csv")
    query_test = pd.read_csv("data/query_test.csv")
    ground_truth = pd.read_csv("data/ground_truth_data.csv")
    
    # Ambil query pertama dari data_test.csv (untuk retrieval eval)
    query_row = data_test.iloc[0]
    query = query_row['query']
    
    # Cari reference answer dari query_test.csv (untuk generation eval)
    ref_row = query_test[query_test['query'] == query]
    if len(ref_row) == 0:
        print(f"ERROR: Query '{query}' tidak ditemukan di query_test.csv")
        print("Queries yang ada di query_test.csv:")
        print(query_test['query'].tolist())
        return
    reference_answer = ref_row.iloc[0]['reference_answer']
    
    print(f"Query (dari data_test.csv): {query}")
    print(f"Reference Answer (dari query_test.csv): {reference_answer[:100]}...\n")
    
    # Retrieve documents
    print("[2] Retrieving documents...")
    retriever = RetrievalSystem(
        chunks_file="data/data_chunk.csv",
        faiss_index_file="data/faiss_index.index",
        model_path="models/sentence_transformer_model"
    )
    
    retrieved = retriever.search(query, top_k=5)
    print(f"Retrieved {len(retrieved)} documents:\n")
    
    for i, doc in enumerate(retrieved, 1):
        print(f"  [{i}] {doc['judul'][:70]}")
        print(f"      Similarity: {doc['similarity']:.4f}")
        print(f"      Text: {doc['chunk_text'][:80]}...")
        print()
    
    # Get ground truth relevance
    print("[3] Ground truth relevance scores (AI Judge)...")
    gt_query = ground_truth[ground_truth['query_idx'] == 0]
    
    # Normalize retrieved titles
    norm_retrieved_titles = [_normalize_title(doc['judul']) for doc in retrieved]
    print(f"Normalized titles: {norm_retrieved_titles}\n")
    
    # Calculate Precision, MAP, nDCG, MRR
    print("[4] RETRIEVAL METRICS CALCULATION (k=5)...")
    
    # Get relevant and highly relevant docs
    relevant_docs = set(gt_query[gt_query['relevance_score'] >= 1]['doc_title'].apply(_normalize_title))
    highly_relevant_docs = set(gt_query[gt_query['relevance_score'] == 2]['doc_title'].apply(_normalize_title))
    
    print(f"Ground truth relevant docs (score>=1): {len(relevant_docs)}")
    print(f"Ground truth highly relevant docs (score==2): {len(highly_relevant_docs)}")
    print()
    
    # Count matches
    relevant_retrieved = sum(1 for title in norm_retrieved_titles if title in relevant_docs)
    
    # PRECISION@5
    precision = relevant_retrieved / len(norm_retrieved_titles) if len(norm_retrieved_titles) > 0 else 0
    print(f"Precision@5 = {relevant_retrieved}/{len(norm_retrieved_titles)} = {precision:.4f}")
    
    # Get relevance scores for each retrieved doc
    relevances = []
    for i, title in enumerate(norm_retrieved_titles, 1):
        doc_gt = gt_query[gt_query['doc_title'].apply(_normalize_title) == title]
        relevance = int(doc_gt['relevance_score'].iloc[0]) if len(doc_gt) > 0 else 0
        relevances.append(relevance)
        print(f"  Position {i}: {title[:50]} -> relevance={relevance}")
    
    print(f"\nRelevance scores: {relevances}\n")
    
    # MAP (Mean Average Precision)
    precisions_at_rel = []
    num_relevant_seen = 0
    for i, rel in enumerate(relevances, 1):
        if rel >= 1:
            num_relevant_seen += 1
            precisions_at_rel.append(num_relevant_seen / i)
            print(f"  At position {i}: relevant found, P={num_relevant_seen}/{i}={precisions_at_rel[-1]:.4f}")
    
    map_score = np.mean(precisions_at_rel) if precisions_at_rel else 0
    print(f"MAP@5 = {map_score:.4f}\n")
    
    # nDCG (normalized Discounted Cumulative Gain)
    def dcg_k(relevances, k):
        rel = np.asarray(relevances)[:k]
        if rel.size:
            return np.sum(rel / np.log2(np.arange(2, rel.size + 2)))
        return 0.0
    
    dcg = dcg_k(relevances, len(relevances))
    print(f"DCG calculation:")
    for i, rel in enumerate(relevances, 1):
        discount = np.log2(i + 1)
        contrib = rel / discount
        print(f"  Position {i}: {rel}/log2({i+1}) = {rel}/{discount:.4f} = {contrib:.4f}")
    print(f"DCG@5 = {dcg:.4f}\n")
    
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_k(ideal_relevances, len(ideal_relevances))
    print(f"Ideal Relevances: {ideal_relevances}")
    print(f"IDCG@5 = {idcg:.4f}\n")
    
    ndcg = dcg / idcg if idcg > 0 else 0
    print(f"nDCG@5 = DCG/IDCG = {dcg:.4f}/{idcg:.4f} = {ndcg:.4f}\n")
    
    # MRR (Mean Reciprocal Rank)
    mrr = 0
    for i, rel in enumerate(relevances, 1):
        if rel >= 1:
            mrr = 1 / i
            print(f"First relevant document at position {i}")
            print(f"MRR@5 = 1/{i} = {mrr:.4f}")
            break
    
    if mrr == 0:
        print("No relevant documents found")
        print(f"MRR@5 = {mrr:.4f}\n")
    
    print("\n" + "-"*80)
    print("[5] GENERATION METRICS CALCULATION...")
    print("-"*80 + "\n")
    
    # Generate answer
    print("Generating answer using LLM...")
    generator = GenerationSystem(model_name="gemma2:9b")
    
    result = generator.generate_answer(
        query=query,
        retrieved_chunks=retrieved,
        max_context_chunks=5
    )
    
    generated_answer = result.get('answer', '')
    print(f"Generated Answer:\n{generated_answer}\n")
    
    contexts = [doc['chunk_text'] for doc in retrieved]
    
    # BERTScore F1
    print("Calculating BERTScore...")
    _, _, f1 = bert_score(
        [generated_answer], [reference_answer],
        lang='id', model_type='xlm-roberta-base', verbose=False
    )
    bertscore_f1 = float(f1[0])
    print(f"BERTScore F1 = {bertscore_f1:.4f}\n")
    
    # Entity Faithfulness
    print("Calculating Entity Faithfulness...")
    nlp = load_spacy_model()
    
    ans_doc = nlp(generated_answer)
    ans_ents = {ent.text.lower() for ent in ans_doc.ents}
    print(f"Entities in answer: {ans_ents}")
    
    ctx_doc = nlp(" ".join(contexts))
    ctx_ents = {ent.text.lower() for ent in ctx_doc.ents}
    print(f"Entities in contexts: {ctx_ents}")
    
    overlap = len(ans_ents & ctx_ents)
    entity_faith = overlap / len(ans_ents) if ans_ents else 1.0
    print(f"Entity Faithfulness = {overlap}/{len(ans_ents)} = {entity_faith:.4f}\n")
    
    # Answer Relevancy
    print("Calculating Answer Relevancy (Cosine Similarity)...")
    embedder = SentenceTransformer("intfloat/multilingual-e5-base")
    
    q_emb = embedder.encode([query], convert_to_numpy=True)[0]
    a_emb = embedder.encode([generated_answer], convert_to_numpy=True)[0]
    
    print(f"Query embedding shape: {q_emb.shape}")
    print(f"Answer embedding shape: {a_emb.shape}")
    print(f"Query embedding (first 5): {q_emb[:5]} ... (last 1): {q_emb[-1:]}")
    print(f"Answer embedding (first 5): {a_emb[:5]} ... (last 1): {a_emb[-1:]}\n")
    
    dot_product = np.dot(q_emb, a_emb)
    q_norm = np.linalg.norm(q_emb)
    a_norm = np.linalg.norm(a_emb)
    
    print(f"Dot product (q·a) = {dot_product:.4f}")
    print(f"||q|| = {q_norm:.4f}")
    print(f"||a|| = {a_norm:.4f}")
    
    answer_rel = dot_product / (q_norm * a_norm) if (q_norm * a_norm) > 0 else 0.0
    print(f"Answer Relevancy = {dot_product:.4f}/({q_norm:.4f}*{a_norm:.4f}) = {answer_rel:.4f}\n")
    
    # Context Coverage
    print("Calculating Context Coverage...")
    ans_doc_lower = nlp(generated_answer.lower())
    ans_words = {
        tok.text for tok in ans_doc_lower
        if not tok.is_stop and not tok.is_punct and len(tok.text) > 2
    }
    print(f"Content words in answer: {ans_words}")
    
    ctx_doc_lower = nlp(" ".join(contexts).lower())
    ctx_words = {tok.text for tok in ctx_doc_lower if not tok.is_punct}
    print(f"Words in contexts: {ctx_words}")
    
    overlap_words = len(ans_words & ctx_words)
    context_cov = overlap_words / len(ans_words) if ans_words else 0.0
    print(f"Context Coverage = {overlap_words}/{len(ans_words)} = {context_cov:.4f}\n")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF METRICS")
    print("="*80)
    print("\nRETRIEVAL METRICS (@k=5):")
    print(f"  Precision:  {precision:.4f}")
    print(f"  MAP:        {map_score:.4f}")
    print(f"  nDCG:       {ndcg:.4f}")
    print(f"  MRR:        {mrr:.4f}")
    
    print("\nGENERATION METRICS:")
    print(f"  BERTScore F1:        {bertscore_f1:.4f}")
    print(f"  Entity Faithfulness: {entity_faith:.4f}")
    print(f"  Answer Relevancy:    {answer_rel:.4f}")
    print(f"  Context Coverage:    {context_cov:.4f}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    eval_machine_learning_detailed()