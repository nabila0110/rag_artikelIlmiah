"""Generate answers from the saved semantic and BM25 candidate files.

The candidate files are the source of truth for both retrieval and generation.
This keeps the two evaluation tracks aligned and makes generation reproducible
without running retrieval again.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from utils.generation import GenerationSystem


REQUIRED_COLUMNS = {"query_idx", "query", "doc_title", "chunk_text", "rank"}


def load_candidates(path: Path, method: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    frame = frame.copy()
    frame["query_idx"] = frame["query_idx"].astype(int)
    frame["rank"] = frame["rank"].astype(int)
    frame["retrieval_method"] = method
    return frame.sort_values(["query_idx", "rank"])


def as_chunks(group: pd.DataFrame, top_k: int) -> List[Dict]:
    chunks = []
    for _, row in group.sort_values("rank").head(top_k).iterrows():
        chunks.append(
            {
                "judul": row["doc_title"],
                "tahun": row.get("tahun", ""),
                "author": row.get("author", ""),
                "section": row.get("section", ""),
                "chunk_text": row["chunk_text"],
                "chunk_id": row.get("chunk_id", ""),
                "url": row.get("url", ""),
                "rank": int(row["rank"]),
            }
        )
    return chunks


def generate_method(
    candidates: pd.DataFrame,
    output: Path,
    model: str,
    top_k: int,
    resume: bool,
) -> pd.DataFrame:
    cache = {}
    if resume and output.exists():
        old = pd.read_csv(output)
        if {"query_idx", "generated_answer"}.issubset(old.columns):
            cache = {
                int(row["query_idx"]): row.to_dict()
                for _, row in old.iterrows()
            }

    generator = GenerationSystem(model_name=model)
    rows = []
    for query_idx, group in candidates.groupby("query_idx", sort=True):
        query = str(group["query"].iloc[0])
        cached = cache.get(int(query_idx))
        if cached is not None:
            rows.append(cached)
            continue

        chunks = as_chunks(group, top_k)
        result = generator.generate_answer(
            query=query,
            retrieved_chunks=chunks,
            max_context_chunks=top_k,
        )
        rows.append(
            {
                "query_idx": int(query_idx),
                "query": query,
                "retrieval_method": str(group["retrieval_method"].iloc[0]),
                "generated_answer": result.get("answer", ""),
                "contexts": json.dumps(chunks, ensure_ascii=False),
                "context_chunk_ids": json.dumps(
                    [chunk["chunk_id"] for chunk in chunks],
                    ensure_ascii=False,
                ),
                "model": result.get("model", model),
            }
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, index=False)

    result_df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output, index=False)
    return result_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic", type=Path, default=Path("data/semantic_candidates.csv"))
    parser.add_argument("--bm25", type=Path, default=Path("data/bm25_candidates.csv"))
    parser.add_argument("--semantic-output", type=Path, default=Path("data/semantic_nlg_results.csv"))
    parser.add_argument("--bm25-output", type=Path, default=Path("data/bm25_nlg_results.csv"))
    parser.add_argument("--model", default="gemma2:9b")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--methods", nargs="+", choices=["semantic", "bm25"], default=["semantic", "bm25"])
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    files = {
        "semantic": (args.semantic, args.semantic_output),
        "bm25": (args.bm25, args.bm25_output),
    }
    for method in args.methods:
        candidates = load_candidates(files[method][0], method)
        result = generate_method(
            candidates,
            files[method][1],
            args.model,
            args.top_k,
            not args.no_resume,
        )
        print(f"{method}: saved {len(result)} answers to {files[method][1]}")


if __name__ == "__main__":
    main()
