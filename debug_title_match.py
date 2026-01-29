import pandas as pd
from utils.retrieval import RetrievalSystem

def normalize(title):
    return str(title).strip().lower() if title is not None else ""

# Load data
gt_df = pd.read_csv('data/ground_truth_data_pruned.csv')
query_df = pd.read_csv('data/data_test.csv')

# Initialize retriever
retriever = RetrievalSystem('data/data_chunk.csv', 'data/faiss_index.index', 'models/sentence_transformer_model')

print("=" * 80)
print("DIAGNOSTIC: Title Matching Analysis")
print("=" * 80)

for idx, row in query_df.head(3).iterrows():
    qid = row.name
    query = row['query']
    
    # Get ground truth
    gt_query = gt_df[gt_df['query_idx'] == qid]
    gt_relevant = gt_query[gt_query['relevance_score'] >= 1]
    gt_titles_norm = set(gt_relevant['doc_title'].apply(normalize))
    
    # Retrieve all
    results = retriever.search(query, top_k=316)
    
    # Dedupe by title
    retrieved_titles = []
    seen = set()
    for chunk in results:
        title_norm = normalize(chunk['judul'])
        if title_norm not in seen:
            retrieved_titles.append(title_norm)
            seen.add(title_norm)
    
    # Find matches
    matched = gt_titles_norm & set(retrieved_titles)
    missed = gt_titles_norm - set(retrieved_titles)
    
    print(f"\nQuery {qid}: {query[:60]}...")
    print(f"  GT relevant docs: {len(gt_titles_norm)}")
    print(f"  Retrieved unique docs: {len(retrieved_titles)}")
    print(f"  Matched: {len(matched)}")
    print(f"  Missed: {len(missed)}")
    
    if missed:
        print(f"\n  Sample missed titles (first 3):")
        for title in list(missed)[:3]:
            print(f"    - {title[:80]}")
    
    if len(retrieved_titles) < 100:
        print(f"\n  ⚠️ WARNING: Only {len(retrieved_titles)} unique docs retrieved from 316 chunks!")

print("\n" + "=" * 80)
print("Checking ground truth titles vs chunk titles...")
print("=" * 80)

chunks_df = pd.read_csv('data/data_chunk.csv')
chunk_titles_norm = set(chunks_df['judul'].apply(normalize).unique())
gt_all_titles_norm = set(gt_df['doc_title'].apply(normalize).unique())

print(f"Unique titles in chunks: {len(chunk_titles_norm)}")
print(f"Unique titles in ground truth: {len(gt_all_titles_norm)}")
print(f"GT titles NOT in chunks: {len(gt_all_titles_norm - chunk_titles_norm)}")

missing_from_chunks = gt_all_titles_norm - chunk_titles_norm
if missing_from_chunks:
    print(f"\nSample GT titles missing from chunks (first 5):")
    for title in list(missing_from_chunks)[:5]:
        print(f"  - {title[:80]}")
