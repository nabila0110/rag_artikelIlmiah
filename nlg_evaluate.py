"""
NLG evaluation using BERTScore + faithfulness metrics:
- BERTSCORE_F1
- ENTITY_FAITHFULNESS (NER overlap between answer and contexts)
- ANSWER_RELEVANCY (semantic similarity question-answer)
- CONTEXT_COVERAGE (content-word overlap answer vs contexts)

Supports two modes:
1) Provide --generated CSV (columns: query, generated_answer, contexts?)
2) Generate on-the-fly using existing utils/retrieval.py and utils/generation.py
"""

import os
import ast
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import spacy
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer

try:
    from utils.retrieval import RetrievalSystem
    from utils.generation import GenerationSystem
except Exception:
    RetrievalSystem = None
    GenerationSystem = None


def _load_spacy_model(name: str = "xx_ent_wiki_sm"):
    """
    Load spaCy model with automatic fallback to multilingual model.
    
    Default: xx_ent_wiki_sm (multilingual, works for Indonesian)
    Note: Indonesian-specific models (id_core_news_sm) not available in spaCy v3.8+
    
    Args:
        name: Model name
              - xx_ent_wiki_sm: Multilingual (RECOMMENDED, always available)
              - xx_sent_ud_sm: Alternative multilingual model
    
    Returns:
        Loaded spaCy model
    """
    # Fallback chain: requested model -> xx_ent_wiki_sm -> xx_sent_ud_sm
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
            # Try to download
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
                    # Last fallback failed
                    raise RuntimeError(
                        f"Failed to load any spaCy model. Tried: {', '.join(fallback_models)}\n\n"
                        f"Install manually:\n"
                        f"  python -m spacy download xx_ent_wiki_sm\n"
                        f"Or:\n"
                        f"  python -m spacy download xx_sent_ud_sm"
                    )
                # Try next fallback
                continue


def _normalize_contexts(raw) -> List[str]:
    contexts: List[str] = []
    if isinstance(raw, dict) and 'cited_references' in raw:
        raw = raw['cited_references']
    if isinstance(raw, str):
        # Try parse list-like string
        try:
            raw_eval = ast.literal_eval(raw)
            raw = raw_eval
        except Exception:
            raw = [raw]
    for item in raw or []:
        if isinstance(item, dict):
            contexts.append(str(item.get('chunk_text', '')))
        else:
            contexts.append(str(item))
    return contexts


class NLGEvaluator:
    def __init__(
        self,
        spacy_model: str = "xx_ent_wiki_sm",
        bert_model: str = "xlm-roberta-base",
        bert_lang: str = "id",
        embedding_model: str = "intfloat/multilingual-e5-base",
    ):
        self.nlp = _load_spacy_model(spacy_model)
        self.bert_model = bert_model
        self.bert_lang = bert_lang
        self.embedder = SentenceTransformer(embedding_model)

    def bert_f1(self, reference: str, generated: str) -> float:
        _, _, f1 = bert_score(
            [generated], [reference], lang=self.bert_lang,
            model_type=self.bert_model, verbose=False
        )
        return float(f1[0])

    def entity_faithfulness(self, answer: str, contexts: List[str]) -> float:
        ans_doc = self.nlp(answer)
        ans_ents = {ent.text.lower() for ent in ans_doc.ents}
        if not ans_ents:
            return 1.0

        ctx_doc = self.nlp(" ".join(contexts))
        ctx_ents = {ent.text.lower() for ent in ctx_doc.ents}
        overlap = len(ans_ents & ctx_ents)
        return overlap / len(ans_ents)

    def answer_relevancy(self, question: str, answer: str) -> float:
        q_emb = self.embedder.encode([question], convert_to_numpy=True)[0]
        a_emb = self.embedder.encode([answer], convert_to_numpy=True)[0]
        denom = np.linalg.norm(q_emb) * np.linalg.norm(a_emb)
        if denom == 0:
            return 0.0
        return float(np.dot(q_emb, a_emb) / denom)

    def context_coverage(self, answer: str, contexts: List[str]) -> float:
        ans_doc = self.nlp(answer.lower())
        ans_words = {
            tok.text for tok in ans_doc
            if not tok.is_stop and not tok.is_punct and len(tok.text) > 2
        }
        if not ans_words:
            return 0.0

        ctx_doc = self.nlp(" ".join(contexts).lower())
        ctx_words = {tok.text for tok in ctx_doc if not tok.is_punct}
        overlap = len(ans_words & ctx_words)
        return overlap / len(ans_words)

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        reference: Optional[str] = None,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if reference:
            metrics['BERTSCORE_F1'] = self.bert_f1(reference, answer)
        metrics['ENTITY_FAITHFULNESS'] = self.entity_faithfulness(answer, contexts)
        metrics['ANSWER_RELEVANCY'] = self.answer_relevancy(question, answer)
        metrics['CONTEXT_COVERAGE'] = self.context_coverage(answer, contexts)
        metrics['ANSWER_LENGTH'] = len(answer.split())
        return metrics


def run_nlg_evaluation(
    reference_file: str,
    generated_file: Optional[str] = None,
    retrieval_system: Optional[RetrievalSystem] = None,
    generation_system: Optional[GenerationSystem] = None,
    output_file: str = "nlg_eval_results.csv",
    top_k: int = 5,
    spacy_model: str = "xx_ent_wiki_sm",
    bert_model: str = "xlm-roberta-base",
    bert_lang: str = "id",
    embedding_model: str = "intfloat/multilingual-e5-base",
):
    if not os.path.exists(reference_file):
        raise FileNotFoundError(f"Reference file not found: {reference_file}")

    ref_df = pd.read_csv(reference_file)
    required_cols = ['query', 'reference_answer']
    if not all(c in ref_df.columns for c in required_cols):
        raise ValueError(f"Reference file must contain columns: {required_cols}")

    evaluator = NLGEvaluator(
        spacy_model=spacy_model,
        bert_model=bert_model,
        bert_lang=bert_lang,
        embedding_model=embedding_model,
    )

    # Prepare generated answers
    if generated_file is None:
        if retrieval_system is None or generation_system is None:
            raise ValueError("Provide retrieval_system & generation_system or a generated_file")
        generated_rows = []
        for _, row in ref_df.iterrows():
            q = row['query']
            retrieved = retrieval_system.search(q, top_k=top_k)
            contexts = _normalize_contexts(retrieved)
            result = generation_system.generate_answer(
                query=q,
                retrieved_chunks=retrieved if isinstance(retrieved, list) else retrieved.get('cited_references', []),
                max_context_chunks=min(top_k, len(retrieved) if isinstance(retrieved, list) else len(retrieved.get('cited_references', [])) if isinstance(retrieved, dict) else top_k)
            )
            generated_rows.append({
                'query': q,
                'generated_answer': result.get('answer', ''),
                'contexts': contexts,
            })
        gen_df = pd.DataFrame(generated_rows)
    else:
        gen_df = pd.read_csv(generated_file)
        if 'query' not in gen_df.columns or 'generated_answer' not in gen_df.columns:
            raise ValueError("generated_file must have columns: query, generated_answer")
        if 'contexts' not in gen_df.columns:
            if retrieval_system is None:
                raise ValueError("contexts missing; provide retrieval_system to re-fetch")
            ctx_rows = []
            for _, row in gen_df.iterrows():
                retrieved = retrieval_system.search(row['query'], top_k=top_k)
                ctx_rows.append({
                    'query': row['query'],
                    'contexts': _normalize_contexts(retrieved),
                })
            gen_df = gen_df.merge(pd.DataFrame(ctx_rows), on='query', how='left')

    results = []
    for _, ref_row in ref_df.iterrows():
        q = ref_row['query']
        ref_ans = ref_row['reference_answer']
        match = gen_df[gen_df['query'] == q]
        if match.empty:
            continue
        gen_ans = match.iloc[0]['generated_answer']
        contexts = match.iloc[0]['contexts']
        if isinstance(contexts, str):
            contexts = _normalize_contexts(contexts)

        metrics = evaluator.evaluate(q, gen_ans, contexts, reference=ref_ans)
        metrics.update({
            'query': q,
            'reference_answer': ref_ans,
            'generated_answer': gen_ans,
        })
        results.append(metrics)

    results_df = pd.DataFrame(results)

    # Summary stats
    summary = {}
    for col in ['BERTSCORE_F1', 'ENTITY_FAITHFULNESS', 'ANSWER_RELEVANCY', 'CONTEXT_COVERAGE']:
        if col in results_df.columns:
            summary[col] = {
                'mean': float(results_df[col].mean()),
                'std': float(results_df[col].std()),
                'min': float(results_df[col].min()),
                'max': float(results_df[col].max()),
            }

    results_df.to_csv(output_file, index=False)
    print("\nEvaluation complete ->", output_file)
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: Mean {v['mean']:.3f} | Std {v['std']:.3f} | Min {v['min']:.3f} | Max {v['max']:.3f}")

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NLG evaluation with BERTScore and faithfulness metrics")
    parser.add_argument("--reference", type=str, default="data/que_test.csv")
    parser.add_argument("--generated", type=str, default=None)
    parser.add_argument("--output", type=str, default="nlg_eval_results.csv")
    parser.add_argument("--chunks", type=str, default="data/data_chunk.csv")
    parser.add_argument("--faiss_index", type=str, default="data/faiss_index.index")
    parser.add_argument("--model_path", type=str, default="models/sentence_transformer_model")
    parser.add_argument("--llm_model", type=str, default="gemma2:9b")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--spacy_model", type=str, default="xx_ent_wiki_sm", 
                        help="spaCy model for NER (xx_ent_wiki_sm multilingual, works for Indonesian)")
    parser.add_argument("--bert_model", type=str, default="xlm-roberta-base")
    parser.add_argument("--bert_lang", type=str, default="id")
    parser.add_argument("--embedding_model", type=str, default="intfloat/multilingual-e5-base")

    args = parser.parse_args()

    retriever = None
    generator = None
    if args.generated is None:
        if RetrievalSystem is None or GenerationSystem is None:
            raise RuntimeError("RetrievalSystem/GenerationSystem not available; provide --generated")
        retriever = RetrievalSystem(args.chunks, args.faiss_index, args.model_path)
        generator = GenerationSystem(model_name=args.llm_model)

    run_nlg_evaluation(
        reference_file=args.reference,
        generated_file=args.generated,
        retrieval_system=retriever,
        generation_system=generator,
        output_file=args.output,
        top_k=args.top_k,
        spacy_model=args.spacy_model,
        bert_model=args.bert_model,
        bert_lang=args.bert_lang,
        embedding_model=args.embedding_model,
    )
