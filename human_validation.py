"""Export selected generated answers for manual factual-correctness review."""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_ids(values):
    if not values:
        raise ValueError("Provide at least one query ID with --query-ids")
    ids = [int(value) for value in values]
    if len(set(ids)) != len(ids):
        raise ValueError("Query IDs must be unique")
    return ids


def export_validation(
    semantic_file: Path,
    bm25_file: Path,
    reference_file: Path,
    query_ids,
    output: Path,
) -> pd.DataFrame:
    reference = pd.read_csv(reference_file)
    required_reference = {"query", "reference_answer"}
    missing = required_reference.difference(reference.columns)
    if missing:
        raise ValueError(f"Missing columns in {reference_file}: {sorted(missing)}")

    frames = []
    for method, path in (("semantic", semantic_file), ("bm25", bm25_file)):
        if not path.exists():
            raise FileNotFoundError(f"Generated-answer file not found: {path}")
        frame = pd.read_csv(path)
        required = {"query_idx", "query", "generated_answer"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        frame = frame[frame["query_idx"].astype(int).isin(query_ids)].copy()
        frame["retrieval_method"] = method
        frames.append(frame)

    selected = pd.concat(frames, ignore_index=True)
    selected["query_idx"] = selected["query_idx"].astype(int)
    selected = selected.merge(
        reference[["query", "reference_answer"]],
        on="query",
        how="left",
        validate="many_to_one",
    )
    selected["human_factual_score"] = ""
    selected["human_comment"] = ""
    selected["human_reviewer"] = ""
    columns = [
        "query_idx",
        "query",
        "retrieval_method",
        "generated_answer",
        "reference_answer",
        "contexts",
        "human_factual_score",
        "human_comment",
        "human_reviewer",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    selected[columns].sort_values(["query_idx", "retrieval_method"]).to_csv(
        output, index=False
    )
    return selected[columns]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-ids", nargs="+", required=True, help="Selected query IDs")
    parser.add_argument("--semantic", type=Path, default=Path("data/semantic_nlg_results.csv"))
    parser.add_argument("--bm25", type=Path, default=Path("data/bm25_nlg_results.csv"))
    parser.add_argument("--reference", type=Path, default=Path("data/query_test.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/human_validation.csv"))
    args = parser.parse_args()
    result = export_validation(
        args.semantic,
        args.bm25,
        args.reference,
        load_ids(args.query_ids),
        args.output,
    )
    print(f"Exported {len(result)} rows for {result['query_idx'].nunique()} queries to {args.output}")


if __name__ == "__main__":
    main()
