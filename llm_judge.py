"""Create LLM-judged retrieval ground truth from saved candidate files."""

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Dict

import ollama
import pandas as pd
from tqdm.auto import tqdm


def parse_judgment(response: str) -> Dict:
    text = response.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse judge response: {response}")
        value = ast.literal_eval(match.group(0))

    score = int(value["relevance_score"])
    if score not in (0, 1, 2):
        raise ValueError(f"Invalid relevance score: {score}")
    return {
        "relevance_score": score,
        "reasoning": str(value.get("reasoning", "")),
        "confidence": float(value.get("confidence", 0.0)),
    }


def judge_candidate(query: str, candidate: pd.Series, model: str) -> Dict:
    prompt = f"""Anda adalah penilai relevansi dokumen untuk evaluasi sistem pencarian.

Pertanyaan pengguna:
{query}

Dokumen kandidat berikut adalah cuplikan hasil retrieval. Jangan menilai dokumen lain di luar cuplikan ini.
Judul: {candidate['doc_title']}
Tahun: {candidate.get('tahun', 'N/A')}
Cuplikan:
{candidate.get('chunk_text', '')}

Beri nilai relevansi terhadap pertanyaan:
0 = tidak relevan
1 = relevan sebagian atau hanya berkaitan dengan topik
2 = sangat relevan dan berpotensi menjawab pertanyaan

Balas HANYA dengan JSON valid:
{{"relevance_score": 0, "reasoning": "alasan singkat", "confidence": 0.0}}
"""
    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={"temperature": 0, "num_predict": 200},
    )
    return parse_judgment(response["response"])


def load_candidates(semantic_file: Path, bm25_file: Path) -> pd.DataFrame:
    frames = []
    for path in (semantic_file, bm25_file):
        if not path.exists():
            raise FileNotFoundError(f"Candidate file not found: {path}")
        frame = pd.read_csv(path)
        required = {"query_idx", "query", "doc_title", "chunk_text", "rank"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        frames.append(frame)

    candidates = pd.concat(frames, ignore_index=True)
    candidates["query_idx"] = candidates["query_idx"].astype(int)
    candidates["doc_title"] = candidates["doc_title"].astype(str)
    # Judge one representative chunk per query/document, retaining both ranks.
    grouped = []
    for (query_idx, title), group in candidates.groupby(["query_idx", "doc_title"], sort=False):
        representative = group.sort_values("rank").iloc[0].copy()
        methods = set(group.get("retrieval_method", pd.Series(dtype=str)).dropna())
        representative["retrieved_by"] = "+".join(sorted(methods))
        representative["semantic_rank"] = ""
        representative["bm25_rank"] = ""
        for _, row in group.iterrows():
            method = row.get("retrieval_method", "")
            if method == "semantic":
                representative["semantic_rank"] = row["rank"]
            elif method == "bm25":
                representative["bm25_rank"] = row["rank"]
        grouped.append(representative)
    return pd.DataFrame(grouped)


def load_cache(path: Path) -> Dict:
    if not path.exists():
        return {}
    cached = pd.read_csv(path)
    return {
        (int(row.query_idx), str(row.doc_title)): row.to_dict()
        for _, row in cached.iterrows()
    }


def run_judging(
    semantic_file: Path,
    bm25_file: Path,
    output_file: Path,
    model: str,
    resume: bool,
) -> pd.DataFrame:
    candidates = load_candidates(semantic_file, bm25_file)
    cache = load_cache(output_file) if resume else {}
    rows = []
    progress = tqdm(
        candidates.iterrows(),
        total=len(candidates),
        desc="Judging retrieval candidates",
        unit="doc",
    )
    for _, candidate in progress:
        progress.set_postfix(
            query=int(candidate["query_idx"]),
            document=str(candidate["doc_title"])[:30],
            refresh=False,
        )
        key = (int(candidate["query_idx"]), str(candidate["doc_title"]))
        judgment = cache.get(key)
        if judgment is None:
            judgment = judge_candidate(str(candidate["query"]), candidate, model)
        else:
            judgment = {
                "relevance_score": int(judgment["relevance_score"]),
                "reasoning": str(judgment.get("reasoning", "")),
                "confidence": float(judgment.get("confidence", 0.0)),
            }
        rows.append(
            {
                "query_idx": int(candidate["query_idx"]),
                "query": candidate["query"],
                "reference_answer": candidate.get("reference_answer", ""),
                "doc_title": candidate["doc_title"],
                "retrieved_by": candidate.get("retrieved_by", ""),
                "semantic_rank": candidate.get("semantic_rank", ""),
                "bm25_rank": candidate.get("bm25_rank", ""),
                "chunk_id": candidate.get("chunk_id", ""),
                "chunk_text": candidate.get("chunk_text", ""),
                "url": candidate.get("url", ""),
                **judgment,
            }
        )
        pd.DataFrame(rows).to_csv(output_file, index=False)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic", type=Path, default=Path("data/semantic_candidates.csv"))
    parser.add_argument("--bm25", type=Path, default=Path("data/bm25_candidates.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/ground_truth_retrieved.csv"))
    parser.add_argument("--judge-model", default="gemma2:9b")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run_judging(args.semantic, args.bm25, args.output, args.judge_model, not args.no_resume)
    print(f"Saved {len(result)} judged candidate rows to {args.output}")


if __name__ == "__main__":
    main()
