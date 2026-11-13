"""Run automated claim-level faithfulness and citation evaluation."""

import argparse
import ast
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List

import ollama
import pandas as pd
from nlg_evaluate import NLGEvaluator, _normalize_contexts


logger = logging.getLogger(__name__)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


def normalize_score(value: object, field: str) -> int:
    """Convert judge scores to the documented 0-2 scale."""
    try:
        score = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be an integer score, got {value!r}") from error
    if score < 0:
        raise ValueError(f"{field} cannot be negative, got {score}")
    if score > 2:
        logger.warning(
            "Normalizing out-of-range %s=%s to 2 for the documented 0-2 scale",
            field,
            score,
        )
        return 2
    return score


def parse_json(response: str) -> Dict:
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(
                f"Could not parse judge response as JSON: {response[:500]}"
            )
        try:
            value = ast.literal_eval(match.group(0))
        except (SyntaxError, ValueError) as error:
            raise ValueError(
                f"Judge response contains incomplete or invalid JSON: {response[:500]}"
            ) from error
    if not isinstance(value, dict):
        raise ValueError("Judge response must be a JSON object")
    return value


def load_contexts(value: object) -> List[Dict]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
    return parsed if isinstance(parsed, list) else []


def judge_answer(row: pd.Series, model: str) -> Dict:
    contexts = load_contexts(row.get("contexts", ""))
    context_text = "\n\n".join(
        f"[{index}] {item.get('judul', '')}\n{item.get('chunk_text', '')}"
        for index, item in enumerate(contexts, 1)
    )
    prompt = f"""Anda adalah evaluator jawaban RAG berbahasa Indonesia.

Pertanyaan:
{row['query']}

Jawaban yang dihasilkan:
{row['generated_answer']}

Konteks retrieved:
{context_text}

Pecah jawaban menjadi klaim atomik. Untuk setiap klaim, nilai:
- factual_correct: 2 jika benar dan didukung konteks, 1 jika sebagian benar,
  0 jika salah atau tidak didukung.
- citation_correct: 2 jika citation [n] menunjuk konteks yang mendukung klaim,
  1 jika dukungan sebagian, 0 jika citation hilang/salah.

Balas HANYA JSON valid dan ringkas dengan bentuk:
{{"claims":[{{"claim":"...", "factual_correct":0, "citation_correct":0}}]}}

PENTING:
- factual_correct dan citation_correct HARUS berupa integer 0, 1, atau 2.
- Jangan gunakan nilai lain, persentase, boolean, atau label teks.
- Jangan sertakan reasoning atau field lain.
"""
    last_response = ""
    normalized = []
    for attempt in range(3):
        response = ollama.generate(
            model=model,
            prompt=prompt
            + f"\nPercobaan {attempt + 1}/3: gunakan hanya skor integer 0, 1, atau 2.",
            format="json",
            options={"temperature": 0, "num_predict": 1200},
        )["response"]
        last_response = response
        try:
            value = parse_json(response)
            claims = value.get("claims")
            if not isinstance(claims, list):
                raise ValueError("Judge response is missing a claims list")

            normalized = []
            for claim in claims:
                if not isinstance(claim, dict):
                    raise ValueError("Each claim judgment must be an object")
                factual = normalize_score(claim["factual_correct"], "factual_correct")
                citation = normalize_score(
                    claim["citation_correct"], "citation_correct"
                )
                normalized.append(
                    {
                        "claim": str(claim.get("claim", "")),
                        "factual_correct": factual,
                        "citation_correct": citation,
                    }
                )
            break
        except (ValueError, KeyError, TypeError) as error:
            if attempt == 2:
                raise ValueError(
                    "Judge returned an invalid response after 3 attempts. "
                    f"Last response: {last_response}"
                ) from error

    total = len(normalized)
    return {
        "claims": json.dumps(normalized, ensure_ascii=False),
        "claim_count": total,
        "faithfulness_score": (
            sum(item["factual_correct"] for item in normalized) / (2 * total)
            if total else 0.0
        ),
        "citation_correctness": (
            sum(item["citation_correct"] for item in normalized) / (2 * total)
            if total else 0.0
        ),
    }


def evaluate_file(path: Path, output: Path, model: str, resume: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Generated-answer file not found: {path}")
    source = pd.read_csv(path)
    required = {"query_idx", "query", "generated_answer", "contexts", "retrieval_method"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    cache = {}
    if resume and output.exists():
        old = pd.read_csv(output)
        cache = {int(row.query_idx): row.to_dict() for _, row in old.iterrows()}

    rows = []
    for _, row in source.iterrows():
        query_idx = int(row["query_idx"])
        judgment = cache.get(query_idx)
        if judgment is None:
            judgment = judge_answer(row, model)
        rows.append({**row.to_dict(), **judgment})
        pd.DataFrame(rows).to_csv(output, index=False)
    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def add_similarity_metrics(
    results: pd.DataFrame,
    reference_by_query: Dict[str, str],
    evaluator: NLGEvaluator,
) -> pd.DataFrame:
    similarity_columns = [
        "BERT_PRECISION",
        "BERT_RECALL",
        "BERTSCORE_F1",
        "ANSWER_LENGTH",
    ]
    results = results.drop(columns=similarity_columns, errors="ignore")
    metrics = []
    for _, row in results.iterrows():
        query = str(row["query"])
        reference = reference_by_query.get(query)
        if reference is None:
            raise ValueError(f"No reference answer found for query: {query}")
        contexts = _normalize_contexts(row.get("contexts", ""))
        metrics.append(
            evaluator.evaluate(
                question=query,
                answer=str(row["generated_answer"]),
                contexts=contexts,
                reference=reference,
            )
        )
    metric_df = pd.DataFrame(metrics)
    return pd.concat(
        [results.reset_index(drop=True), metric_df.reset_index(drop=True)],
        axis=1,
    )


def summarize_results(results: pd.DataFrame) -> Dict[str, float]:
    return {
        "queries": int(len(results)),
        "claims": int(pd.to_numeric(results["claim_count"]).sum()),
        "bert_precision_percent": float(
            pd.to_numeric(results["BERT_PRECISION"]).mean() * 100
        ),
        "bert_recall_percent": float(
            pd.to_numeric(results["BERT_RECALL"]).mean() * 100
        ),
        "bertscore_f1_percent": float(
            pd.to_numeric(results["BERTSCORE_F1"]).mean() * 100
        ),
        "faithfulness_percent": float(
            pd.to_numeric(results["faithfulness_score"]).mean() * 100
        ),
        "citation_correctness_percent": float(
            pd.to_numeric(results["citation_correctness"]).mean() * 100
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("query_set_50.csv"),
        help="Reference CSV with query_bahasa_indonesia and reference_answer_bahasa_indonesia",
    )
    parser.add_argument("--semantic", type=Path, default=Path("data/semantic_nlg_results.csv"))
    parser.add_argument("--bm25", type=Path, default=Path("data/bm25_nlg_results.csv"))
    parser.add_argument("--semantic-output", type=Path, default=Path("data/semantic_generation_eval.csv"))
    parser.add_argument("--bm25-output", type=Path, default=Path("data/bm25_generation_eval.csv"))
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/generation_eval_summary.csv"),
    )
    parser.add_argument("--judge-model", default="gemma2:9b")
    parser.add_argument("--bert-model", default="xlm-roberta-base")
    parser.add_argument("--bert-lang", default="id")
    parser.add_argument("--methods", nargs="+", choices=["semantic", "bm25"], default=["semantic", "bm25"])
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if not args.reference.exists():
        raise FileNotFoundError(f"Reference file not found: {args.reference}")
    reference_df = pd.read_csv(args.reference)
    reference_columns = {
        "query": "query",
        "reference_answer": "reference_answer",
        "query_bahasa_indonesia": "query",
        "reference_answer_bahasa_indonesia": "reference_answer",
    }
    query_column = next(
        (column for column in reference_columns if column in reference_df.columns),
        None,
    )
    answer_column = (
        "reference_answer"
        if "reference_answer" in reference_df.columns
        else "reference_answer_bahasa_indonesia"
    )
    if query_column is None or answer_column not in reference_df.columns:
        raise ValueError(
            "Reference file must contain query/reference_answer or "
            "query_bahasa_indonesia/reference_answer_bahasa_indonesia"
        )
    reference_by_query = dict(
        zip(
            reference_df[query_column].astype(str),
            reference_df[answer_column].fillna("").astype(str),
        )
    )
    bert_evaluator = NLGEvaluator(
        bert_model=args.bert_model,
        bert_lang=args.bert_lang,
        idf_sents=list(reference_by_query.values()),
    )
    files = {
        "semantic": (args.semantic, args.semantic_output),
        "bm25": (args.bm25, args.bm25_output),
    }
    summaries = []
    for method in args.methods:
        result = evaluate_file(files[method][0], files[method][1], args.judge_model, not args.no_resume)
        result = add_similarity_metrics(result, reference_by_query, bert_evaluator)
        result.to_csv(files[method][1], index=False)
        summary = summarize_results(result)
        summaries.append({"method": method, **summary})
        print(f"{method}: evaluated {len(result)} answers -> {files[method][1]}")
        print(f"  Claims evaluated: {summary['claims']}")
        print(f"  BERTScore precision: {summary['bert_precision_percent']:.2f}%")
        print(f"  BERTScore recall: {summary['bert_recall_percent']:.2f}%")
        print(f"  BERTScore F1: {summary['bertscore_f1_percent']:.2f}%")
        print(f"  Mean faithfulness: {summary['faithfulness_percent']:.2f}%")
        print(
            "  Mean citation correctness: "
            f"{summary['citation_correctness_percent']:.2f}%"
        )

    summary_df = pd.DataFrame(summaries)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.summary_output, index=False)
    print(f"Summary saved to {args.summary_output}")
    print("\nGENERATION EVALUATION SUMMARY")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
