"""Evaluate saved semantic and BM25 retrieval candidates.

This script only reads retrieval candidates and LLM judgments. It does not
run a retriever, call Ollama, or create ground-truth labels.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ir_evaluate import _normalize_title, evaluate_retrieval


def load_ground_truth(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")
    ground_truth = pd.read_csv(path)
    required = {"query_idx", "doc_title", "relevance_score"}
    missing = required.difference(ground_truth.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    ground_truth = ground_truth.copy()
    ground_truth["query_idx"] = ground_truth["query_idx"].astype(int)
    ground_truth["doc_title_norm"] = ground_truth["doc_title"].apply(_normalize_title)
    return ground_truth


def load_candidates(path: Path, method: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")
    candidates = pd.read_csv(path)
    required = {"query_idx", "query", "doc_title", "rank", "chunk_text"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    candidates = candidates.copy()
    candidates["query_idx"] = candidates["query_idx"].astype(int)
    candidates["rank"] = candidates["rank"].astype(int)
    candidates["retrieval_method"] = method
    return candidates


def as_retrieved_chunks(candidates: pd.DataFrame, query_idx: int) -> List[Dict]:
    query_candidates = candidates[candidates["query_idx"] == query_idx].sort_values("rank")
    return [
        {
            "judul": row["doc_title"],
            "chunk_text": row["chunk_text"],
            "chunk_id": row.get("chunk_id", ""),
            "similarity": row.get("retrieval_score", 0.0),
        }
        for _, row in query_candidates.iterrows()
    ]


def evaluate_method(
    method: str,
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
    k_values: List[int],
) -> pd.DataFrame:
    query_rows = candidates[["query_idx", "query"]].drop_duplicates().sort_values("query_idx")
    rows = []
    for _, query_row in query_rows.iterrows():
        query_idx = int(query_row["query_idx"])
        retrieved_chunks = as_retrieved_chunks(candidates, query_idx)
        for k in k_values:
            metrics = evaluate_retrieval(
                query_idx=query_idx,
                retrieved_chunks=retrieved_chunks,
                ground_truth_df=ground_truth,
                k=k,
            )
            rows.append(
                {
                    "method": method,
                    "query_idx": query_idx,
                    "query": query_row["query"],
                    "k": k,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def print_summary(results: pd.DataFrame) -> None:
    for (method, k), group in results.groupby(["method", "k"], sort=True):
        print(f"\n{method.upper()} @k={k} ({len(group)} queries)")
        for metric in ("Precision", "MRR", "MAP", "nDCG"):
            print(f"  {metric}: {group[metric].mean():.4f} +/- {group[metric].std():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=Path("data/ground_truth_retrieved.csv"))
    parser.add_argument("--semantic", type=Path, default=Path("data/semantic_candidates.csv"))
    parser.add_argument("--bm25", type=Path, default=Path("data/bm25_candidates.csv"))
    parser.add_argument("--output", type=Path, default=Path("eval_results.csv"))
    parser.add_argument("--k", type=int, nargs="*", default=[5, 10])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["semantic", "bm25"],
        default=["semantic", "bm25"],
        help="Methods to score; both are evaluated by default.",
    )
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.ground_truth)
    files = {"semantic": args.semantic, "bm25": args.bm25}
    results = []
    for method in args.methods:
        candidates = load_candidates(files[method], method)
        results.append(evaluate_method(method, candidates, ground_truth, args.k))

    result_df = pd.concat(results, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)
    print_summary(result_df)
    print(f"\nSaved evaluation metrics to {args.output}")


if __name__ == "__main__":
    main()
