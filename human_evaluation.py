"""Calculate human retrieval and semantic-generation evaluation results."""

import argparse
import math
import re
from pathlib import Path

import pandas as pd


def parse_score(value, labels):
    if pd.isna(value):
        return None
    text = str(value).strip()
    for score, names in labels.items():
        if text in names:
            return score
    match = re.search(r"\b([012])\b", text)
    return int(match.group(1)) if match else None


def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_header(header):
    if not isinstance(header, str):
        return None, None
    parts = re.split(r"\n\s*=\s*", header, maxsplit=1)
    if len(parts) != 2:
        return None, None
    return normalize_text(parts[0]), normalize_text(parts[1])


def dcg(relevances):
    return sum((2 ** grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(relevances, 1))


def ndcg(relevances, ideal_relevances):
    ideal = dcg(sorted(ideal_relevances, reverse=True)[:len(relevances)])
    return dcg(relevances) / ideal if ideal else 0.0


def retrieval_metrics(relevances, k=5, binary_threshold=1):
    relevances = relevances[:k]
    if len(relevances) < k:
        relevances = relevances + [0] * (k - len(relevances))
    binary = [1 if grade >= binary_threshold else 0 for grade in relevances]
    precision = sum(binary) / len(binary) if binary else 0.0

    first = next((rank for rank, grade in enumerate(binary, 1) if grade == 1), None)
    mrr = 1.0 / first if first else 0.0

    hits = 0
    precision_sum = 0.0
    for rank, grade in enumerate(binary, 1):
        if grade:
            hits += 1
            precision_sum += hits / rank
    relevant_count = sum(binary)
    ap = precision_sum / relevant_count if relevant_count else 0.0

    ideal = sorted(relevances, reverse=True)
    score = ndcg(relevances, ideal)
    return {
        "Precision": precision,
        "MRR": mrr,
        "MAP": ap,
        "nDCG": score,
    }


def load_candidates(path):
    frame = pd.read_csv(path)
    required = {"query_idx", "query", "rank", "chunk_text"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    frame = frame.copy()
    frame["query_idx"] = frame["query_idx"].astype(int)
    frame["rank"] = frame["rank"].astype(int)
    frame["_query"] = frame["query"].map(normalize_text)
    frame["_chunk"] = frame["chunk_text"].map(normalize_text)
    return frame


def evaluate_human(
    human_file,
    semantic_candidates,
    bm25_candidates,
    generation_output,
    retrieval_output,
    generation_summary_output,
    retrieval_summary_output,
):
    human = pd.read_csv(human_file)
    semantic = load_candidates(semantic_candidates)
    bm25 = load_candidates(bm25_candidates)
    generation = pd.read_csv(generation_output)

    generated_lookup = {}
    for _, row in generation.iterrows():
        if int(row["query_idx"]) not in generated_lookup:
            generated_lookup[int(row["query_idx"])] = normalize_text(row["generated_answer"])

    retrieval_labels = {
        0: ("0 - Tidak Relevan",),
        1: ("1 - Relevan",),
        2: ("2 - Sangat Relevan",),
    }
    factual_labels = {
        0: ("Tidak Tepat",),
        1: ("Tepat",),
        2: ("Sangat Tepat",),
    }

    retrieval_rows = []
    generation_rows = []
    retrieval_positions = {}

    for column in human.columns[3:]:
        query, item = parse_header(column)
        if query is None:
            continue

        position = retrieval_positions.setdefault(query, 0)
        target_method = "semantic" if position < 5 else "bm25"
        semantic_rows = semantic[
            (semantic["_query"] == query)
            & (semantic["_chunk"] == item)
        ]
        bm25_rows = bm25[
            (bm25["_query"] == query)
            & (bm25["_chunk"] == item)
        ]
        if not semantic_rows.empty and bm25_rows.empty:
            candidate_rows = semantic_rows
            method = "semantic"
        elif semantic_rows.empty and not bm25_rows.empty:
            candidate_rows = bm25_rows
            method = "bm25"
        elif not semantic_rows.empty and not bm25_rows.empty:
            candidate_rows = semantic_rows if target_method == "semantic" else bm25_rows
            method = target_method
        else:
            candidate_rows = pd.DataFrame()
            method = target_method

        if candidate_rows.empty:
            candidate_source = semantic if target_method == "semantic" else bm25
            target_candidates = candidate_source[
                candidate_source["_query"] == query
            ].sort_values("rank").head(10)
            fuzzy_matches = []
            for _, candidate_row in target_candidates.iterrows():
                candidate_text = normalize_text(candidate_row["_chunk"])
                if item in candidate_text or candidate_text in item:
                    smaller = min(len(item), len(candidate_text))
                    larger = max(len(item), len(candidate_text))
                    overlap_ratio = smaller / larger if larger else 0.0
                    fuzzy_matches.append((overlap_ratio, candidate_row))
            if fuzzy_matches:
                overlap_ratio, candidate_row = max(
                    fuzzy_matches,
                    key=lambda match: match[0],
                )
                if overlap_ratio >= 0.80:
                    candidate_rows = pd.DataFrame([candidate_row])

        if candidate_rows.empty:
            generated_for_query = generation[
                generation["query"].map(normalize_text) == query
            ]
            is_generation_header = any(
                normalize_text(answer_row["generated_answer"]) == item
                for _, answer_row in generated_for_query.iterrows()
            )
            if not is_generation_header:
                retrieval_positions[query] = position + 1

        if not candidate_rows.empty:
            candidate = candidate_rows.iloc[0]
            retrieval_positions[query] = position + 1
            judge_scores = [
                parse_score(human.iloc[row][column], retrieval_labels)
                for row in range(len(human))
            ]
            judge_scores = [score for score in judge_scores if score is not None]
            if judge_scores:
                retrieval_rows.append({
                    "query_idx": int(candidate["query_idx"]),
                    "query": query,
                    "method": method,
                    "rank": int(candidate["rank"]),
                    "chunk_id": str(candidate.get("chunk_id", "")),
                    "chunk_text": str(candidate["chunk_text"]),
                    "human_score_mean": sum(judge_scores) / len(judge_scores),
                    "human_score_median": pd.Series(judge_scores).median(),
                    "judge_count": len(judge_scores),
                })
            continue

        query_rows = generation[
            generation["query"].map(normalize_text) == query
        ]
        if query_rows.empty:
            continue

        for _, answer_row in query_rows.iterrows():
            generated_answer = normalize_text(answer_row["generated_answer"])
            if generated_answer == item:
                scores = [
                    parse_score(human.iloc[row][column], factual_labels)
                    for row in range(len(human))
                ]
                scores = [score for score in scores if score is not None]
                if scores:
                    generation_rows.append({
                        "query_idx": int(answer_row["query_idx"]),
                        "query": query,
                        "method": "semantic",
                        "generated_answer": str(answer_row["generated_answer"]),
                        "human_score_mean": sum(scores) / len(scores),
                        "human_score_median": pd.Series(scores).median(),
                        "human_correct_binary": int(
                            sum(scores) / len(scores) >= 1.0
                        ),
                        "human_correct_judge_count": sum(score >= 1 for score in scores),
                        "human_total_judge_count": len(scores),
                        "judge_count": len(scores),
                    })

    retrieval_df = pd.DataFrame(retrieval_rows)
    if retrieval_df.empty:
        raise ValueError("No retrieval items matched semantic/bm25 candidate files")

    retrieval_output.parent.mkdir(parents=True, exist_ok=True)
    retrieval_df.to_csv(retrieval_output, index=False)

    retrieval_per_query = []
    retrieval_rows_by_key = {}
    for (method, query_idx), group in retrieval_df.groupby(["method", "query_idx"]):
        group = group.sort_values("rank")
        grades = list(group.sort_values("rank")["human_score_median"].fillna(0).astype(int))
        metrics = retrieval_metrics(grades, k=len(grades), binary_threshold=1)
        retrieval_per_query.append({
            "method": method,
            "query_idx": int(query_idx),
            "query": str(group["query"].iloc[0]),
            "matched_chunks": len(group),
            "chunk_coverage_percent": len(group) / 5 * 100,
            **metrics,
        })
        retrieval_rows_by_key[(method, int(query_idx))] = group

    retrieval_query_df = pd.DataFrame(retrieval_per_query)
    retrieval_query_df.to_csv(retrieval_output.with_name("human_retrieval_per_query.csv"), index=False)

    retrieval_summary_rows = []
    for method, group in retrieval_query_df.groupby("method"):
        retrieval_summary_rows.append({
            "method": method,
            "queries": len(group),
            "matched_chunks": int(group["matched_chunks"].sum()),
            "chunk_coverage_percent": group["chunk_coverage_percent"].mean(),
            "human_precision": group["Precision"].mean(),
            "human_mrr": group["MRR"].mean(),
            "human_map": group["MAP"].mean(),
            "human_ndcg": group["nDCG"].mean(),
        })
    retrieval_summary_df = pd.DataFrame(retrieval_summary_rows)
    retrieval_summary_output.parent.mkdir(parents=True, exist_ok=True)
    retrieval_summary_df.to_csv(retrieval_summary_output, index=False)

    generation_df = pd.DataFrame(generation_rows)
    generation_output.parent.mkdir(parents=True, exist_ok=True)
    generation_df.to_csv(generation_output.with_name("human_generation_per_query.csv"), index=False)

    generation_summary_df = pd.DataFrame()
    if not generation_df.empty:
        human_correct = generation_df["human_correct_binary"]
        generation_summary_df = pd.DataFrame([{
            "method": "semantic",
            "queries": len(generation_df),
            "human_factual_accuracy_percent": human_correct.mean() * 100,
            "human_judge_factual_accuracy_percent": (
                generation_df["human_correct_judge_count"].sum()
                / generation_df["human_total_judge_count"].sum()
                * 100
            ),
            "human_quality_score_mean": generation_df["human_score_mean"].mean(),
            "human_quality_score_percent": generation_df["human_score_mean"].mean() / 2 * 100,
            "Tidak Tepat": (generation_df["human_score_median"] == 0).sum(),
            "Tepat": (generation_df["human_score_median"] == 1).sum(),
            "Sangat Tepat": (generation_df["human_score_median"] == 2).sum(),
        }])
        generation_summary_output.parent.mkdir(parents=True, exist_ok=True)
        generation_summary_df.to_csv(generation_summary_output, index=False)

    print("\nHUMAN RETRIEVAL EVALUATION")
    for _, row in retrieval_summary_df.iterrows():
        print(f"\n{row['method'].upper()}")
        print(f"  Queries:      {int(row['queries'])}")
        print(f"  Matched chunks: {int(row['matched_chunks'])} / {int(row['queries'] * 5)}")
        print(f"  Chunk coverage: {row['chunk_coverage_percent']:.2f}%")
        print(f"  Human Precision: {row['human_precision']:.4%}")
        print(f"  Human MRR:       {row['human_mrr']:.4%}")
        print(f"  Human MAP:       {row['human_map']:.4%}")
        print(f"  Human nDCG:      {row['human_ndcg']:.4%}")

    print("\nHUMAN SEMANTIC GENERATION EVALUATION")
    if generation_summary_df.empty:
        print("  No generated answers matched the human evaluation file.")
    else:
        row = generation_summary_df.iloc[0]
        print(f"  Queries:                 {int(row['queries'])}")
        print(f"  Factual accuracy:        {row['human_judge_factual_accuracy_percent']:.2f}%")
        print(f"  Query-level accuracy:    {row['human_factual_accuracy_percent']:.2f}%")
        print(f"  Mean quality score:      {row['human_quality_score_mean']:.4f} / 2")
        print(f"  Quality score:           {row['human_quality_score_percent']:.2f}%")

    print(f"\nSaved retrieval details -> {retrieval_output}")
    print(f"Saved retrieval per-query -> {retrieval_output.with_name('human_retrieval_per_query.csv')}")
    print(f"Saved retrieval summary -> {retrieval_summary_output}")
    print(f"Saved generation per-query -> {generation_output.with_name('human_generation_per_query.csv')}")
    print(f"Saved generation summary -> {generation_summary_output}")

    return retrieval_summary_df, generation_summary_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", type=Path, default=Path("data/Human Evaluation.csv"))
    parser.add_argument("--semantic-candidates", type=Path, default=Path("data/semantic_candidates.csv"))
    parser.add_argument("--bm25-candidates", type=Path, default=Path("data/bm25_candidates.csv"))
    parser.add_argument("--generation", type=Path, default=Path("data/semantic_nlg_results.csv"))
    parser.add_argument("--retrieval-output", type=Path, default=Path("data/human_retrieval_results.csv"))
    parser.add_argument("--retrieval-summary-output", type=Path, default=Path("data/human_retrieval_summary.csv"))
    parser.add_argument("--generation-summary-output", type=Path, default=Path("data/human_generation_summary.csv"))
    args = parser.parse_args()

    evaluate_human(
        human_file=args.human,
        semantic_candidates=args.semantic_candidates,
        bm25_candidates=args.bm25_candidates,
        generation_output=args.generation,
        retrieval_output=args.retrieval_output,
        generation_summary_output=args.generation_summary_output,
        retrieval_summary_output=args.retrieval_summary_output,
    )


if __name__ == "__main__":
    main()
