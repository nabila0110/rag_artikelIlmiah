"""Build BM25 retrieval candidates and save them for later evaluation."""

import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", str(text).lower(), flags=re.UNICODE)


class BM25Retriever:
    def __init__(self, chunks_df: pd.DataFrame):
        self.chunks_df = chunks_df.reset_index(drop=True)
        corpus = self.chunks_df["chunk_text"].fillna("").astype(str)
        self.bm25 = BM25Okapi([tokenize(text) for text in corpus])

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked_indices = sorted(
            range(len(scores)), key=lambda index: scores[index], reverse=True
        )

        results = []
        seen_titles = set()
        for index in ranked_indices:
            row = self.chunks_df.iloc[index]
            title = str(row.get("judul", "")).strip()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            results.append(
                {
                    "doc_title": title,
                    "chunk_id": row.get("chunk_idx", f"chunk_{index}"),
                    "chunk_text": str(row.get("chunk_text", "")),
                    "author": row.get("first_author", "unknown"),
                    "tahun": row.get("tahun_terbit", "N/A"),
                    "url": row.get("url", "#"),
                    "section": row.get("chunk_section", "unknown"),
                    "retrieval_score": float(scores[index]),
                    "rank": len(results) + 1,
                    "retrieval_method": "bm25",
                }
            )
            if len(results) >= top_k:
                break
        return results


def load_queries(path: Path) -> pd.DataFrame:
    queries = pd.read_csv(path)
    queries = queries.rename(
        columns={
            "no": "query_idx",
            "query_bahasa_indonesia": "query",
            "reference_answer_bahasa_indonesia": "reference_answer",
        }
    )
    required = {"query", "reference_answer"}
    missing = required.difference(queries.columns)
    if missing:
        raise ValueError(f"Missing query columns in {path}: {sorted(missing)}")
    if "query_idx" not in queries.columns:
        queries.insert(0, "query_idx", range(len(queries)))
    return queries


def run_bm25(query_file: Path, chunks_file: Path, output_file: Path, top_k: int) -> pd.DataFrame:
    queries = load_queries(query_file)
    retriever = BM25Retriever(pd.read_csv(chunks_file))
    rows = []
    for _, query_row in queries.iterrows():
        for result in retriever.search(str(query_row["query"]), top_k=top_k):
            rows.append(
                {
                    "query_idx": int(query_row["query_idx"]),
                    "query": query_row["query"],
                    "reference_answer": query_row["reference_answer"],
                    **result,
                }
            )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_file, index=False)
    return result_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=Path("query_set_50.csv"))
    parser.add_argument("--chunks", type=Path, default=Path("data/data_chunk.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/bm25_candidates.csv"))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    result = run_bm25(args.queries, args.chunks, args.output, args.top_k)
    print(f"Saved {len(result)} BM25 candidate rows to {args.output}")


if __name__ == "__main__":
    main()
