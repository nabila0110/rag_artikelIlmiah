"""
NLG evaluation using BERTScore + faithfulness metrics:
- BERTSCORE_F1
- ENTITY_FAITHFULNESS (NER overlap between answer and contexts)
- CONTEXT_COVERAGE (content-word overlap answer vs contexts)

Supports two modes:
1) Provide --generated CSV (columns: query, generated_answer, contexts?)
2) Generate on-the-fly using existing utils/retrieval.py and utils/generation.py
"""

import os
import ast
from typing import List, Dict, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import spacy
from bert_score import BERTScorer
from bert_score import utils as bert_utils

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
        idf_sents: Optional[List[str]] = None,
    ):
        self.nlp = _load_spacy_model(spacy_model)
        self.bert_model = bert_model
        self.bert_lang = bert_lang
        self._idf_sents = [s for s in (idf_sents or []) if s]
        self._bert_scorer = BERTScorer(
            model_type=self.bert_model,
            lang=self.bert_lang,
            idf=bool(self._idf_sents),
            idf_sents=self._idf_sents if self._idf_sents else None,
            rescale_with_baseline=False,
        )
        self._idf_dict = None
        self._df_dict = None
        self._idf_N = 0
        if self._idf_sents:
            tokenizer = self._bert_scorer._tokenizer
            self._idf_dict = bert_utils.get_idf_dict(self._idf_sents, tokenizer)
            self._df_dict = defaultdict(int)
            for sent in self._idf_sents:
                token_ids = set(bert_utils.sent_encode(tokenizer, sent))
                for tid in token_ids:
                    self._df_dict[tid] += 1
            self._idf_N = len(self._idf_sents)
            if tokenizer.sep_token_id is not None:
                self._idf_dict[tokenizer.sep_token_id] = 0.0
            if tokenizer.cls_token_id is not None:
                self._idf_dict[tokenizer.cls_token_id] = 0.0

    def _get_bert_debug_model(self):
        return self._bert_scorer._tokenizer, self._bert_scorer._model

    def _bert_prf_debug(self, reference: str, generated: str) -> Dict[str, float]:
        import torch

        tokenizer, model = self._get_bert_debug_model()
        device = self._bert_scorer.device

        if self._idf_dict is not None:
            idf_dict = self._idf_dict
        else:
            idf_dict = defaultdict(lambda: 1.0)
            if tokenizer.sep_token_id is not None:
                idf_dict[tokenizer.sep_token_id] = 0.0
            if tokenizer.cls_token_id is not None:
                idf_dict[tokenizer.cls_token_id] = 0.0

        ref_ids = bert_utils.sent_encode(tokenizer, reference)
        gen_ids = bert_utils.sent_encode(tokenizer, generated)
        ref_tokens = tokenizer.convert_ids_to_tokens(ref_ids)
        gen_tokens = tokenizer.convert_ids_to_tokens(gen_ids)

        ref_emb, ref_mask, ref_idf = bert_utils.get_bert_embedding(
            [reference], model, tokenizer, idf_dict, batch_size=1, device=device, all_layers=False
        )
        gen_emb, gen_mask, gen_idf = bert_utils.get_bert_embedding(
            [generated], model, tokenizer, idf_dict, batch_size=1, device=device, all_layers=False
        )

        # Same normalization path as bert_score.utils.greedy_cos_idf
        ref_emb = ref_emb / ref_emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        gen_emb = gen_emb / gen_emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        # Raw cosine matrix: B x hyp_len x ref_len
        sim_raw = torch.bmm(gen_emb, ref_emb.transpose(1, 2))
        masks = torch.bmm(gen_mask.unsqueeze(2).float(), ref_mask.unsqueeze(1).float())
        sim_masked = sim_raw * masks

        gen_len = int(gen_mask[0].sum().item())
        ref_len = int(ref_mask[0].sum().item())

        gen_emb_vecs = gen_emb[0, :gen_len].detach().cpu().tolist()
        ref_emb_vecs = ref_emb[0, :ref_len].detach().cpu().tolist()

        gen_df = [int(self._df_dict.get(tid, 0)) if self._df_dict is not None else 0 for tid in gen_ids[:gen_len]]
        ref_df = [int(self._df_dict.get(tid, 0)) if self._df_dict is not None else 0 for tid in ref_ids[:ref_len]]
        gen_idf_raw = [float(idf_dict.get(tid, 1.0)) for tid in gen_ids[:gen_len]]
        ref_idf_raw = [float(idf_dict.get(tid, 1.0)) for tid in ref_ids[:ref_len]]

        sim_view = sim_masked[0, :gen_len, :ref_len]
        p_max_obj = sim_view.max(dim=1)
        r_max_obj = sim_view.max(dim=0)

        p_max = p_max_obj.values
        r_max = r_max_obj.values
        p_argmax_idx = p_max_obj.indices
        r_argmax_idx = r_max_obj.indices

        # Same weighted precision/recall as bert_score.utils.greedy_cos_idf
        gen_idf_norm = gen_idf / gen_idf.sum(dim=1, keepdim=True).clamp(min=1e-12)
        ref_idf_norm = ref_idf / ref_idf.sum(dim=1, keepdim=True).clamp(min=1e-12)

        precision = float((p_max * gen_idf_norm[0, :gen_len].to(p_max.device)).sum().item())
        recall = float((r_max * ref_idf_norm[0, :ref_len].to(r_max.device)).sum().item())
        denom = precision + recall
        f1 = float((2 * precision * recall / denom) if denom > 0 else 0.0)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ref_tokens": ref_tokens[:ref_len],
            "gen_tokens": gen_tokens[:gen_len],
            "p_max": [float(v) for v in p_max.tolist()],
            "r_max": [float(v) for v in r_max.tolist()],
            "p_argmax_idx": [int(v) for v in p_argmax_idx.tolist()],
            "r_argmax_idx": [int(v) for v in r_argmax_idx.tolist()],
            "gen_idf_norm": [float(v) for v in gen_idf_norm[0, :gen_len].tolist()],
            "ref_idf_norm": [float(v) for v in ref_idf_norm[0, :ref_len].tolist()],
            "gen_idf_raw": gen_idf_raw,
            "ref_idf_raw": ref_idf_raw,
            "gen_df": gen_df,
            "ref_df": ref_df,
            "idf_N": self._idf_N,
            "sim_raw_matrix": [[float(x) for x in row] for row in sim_raw[0, :gen_len, :ref_len].tolist()],
            "sim_matrix": [[float(x) for x in row] for row in sim_view.tolist()],
            "mask_matrix": [[float(x) for x in row] for row in masks[0, :gen_len, :ref_len].tolist()],
            "gen_emb_vecs": gen_emb_vecs,
            "ref_emb_vecs": ref_emb_vecs,
        }

    def bert_f1(self, reference: str, generated: str, verbose: bool = False) -> tuple:
        """Calculate BERT Precision, Recall, and F1 Score"""
        # Use one scorer instance so official score path and debug path are aligned.
        p, r, f1 = self._bert_scorer.score([generated], [reference], verbose=False)
        bert_precision = float(p[0])
        bert_recall = float(r[0])
        f1_score = float(f1[0])
        
        if verbose:
            debug = self._bert_prf_debug(reference, generated)
            def _short_list(items, head=8, tail=1):
                if len(items) <= head + tail:
                    return items
                return items[:head] + ["..."] + items[-tail:]

            def _short_tok(tok: str, max_len: int = 10):
                return tok if len(tok) <= max_len else tok[:max_len-1] + "…"

            def _print_matrix_preview(sim_matrix, gen_tokens, ref_tokens, max_rows=8, max_cols=8):
                if not sim_matrix:
                    print("  Cosine matrix: empty")
                    return

                def _preview_indices(total: int, head_count: int) -> List[int]:
                    if total <= head_count + 1:
                        return list(range(total))
                    return list(range(head_count)) + [total - 1]

                total_rows = len(sim_matrix)
                total_cols = len(sim_matrix[0])
                row_head = max(1, min(max_rows - 1, total_rows))
                col_head = max(1, min(max_cols - 1, total_cols))

                row_idx = _preview_indices(total_rows, row_head)
                col_idx = _preview_indices(total_cols, col_head)

                print(f"  Cosine matrix preview ({len(row_idx)}x{len(col_idx)}) | rows=generated, cols=reference")
                header_tokens = [_short_tok(ref_tokens[j]) for j in col_idx]
                print("    ref→    " + " ".join([f"{t:>10}" for t in header_tokens]))

                for i in row_idx:
                    row_token = _short_tok(gen_tokens[i]) if i < len(gen_tokens) else f"g{i}"
                    vals = [f"{sim_matrix[i][j]:10.4f}" for j in col_idx]
                    print(f"    {_short_tok(row_token, 8):>8} " + " ".join(vals))

                if total_rows > len(row_idx) or total_cols > len(col_idx):
                    print("    ... (showing first rows/cols and last row/col)")

            print(f"\n[BERTScore F1]")
            print(f"  Reference: {reference[:100]}...")
            print(f"  Generated: {generated[:100]}...")
            print(f"  BERT Precision = {bert_precision:.4f}")
            print(f"  BERT Recall    = {bert_recall:.4f}")
            print(f"  F1 = 2 × P × R / (P + R)")
            print(f"  F1 = 2 × {bert_precision:.4f} × {bert_recall:.4f} / ({bert_precision:.4f} + {bert_recall:.4f})")
            print(f"  F1 = {f1_score:.4f}")
            print(f"\n  [Token-level calculation]")
            print(f"  Tokens (generated): {_short_list(debug['gen_tokens'])}")
            print(f"  Tokens (reference): {_short_list(debug['ref_tokens'])}")
            print(f"  Formula matrix: S_raw = E_gen_norm × E_ref_norm^T")
            print(f"  Masked matrix: S = S_raw ⊙ M (M from attention masks)")
            print(f"  IDF raw (gen): {_short_list([f'{v:.4f}' for v in debug['gen_idf_raw']])}")
            print(f"  IDF raw (ref): {_short_list([f'{v:.4f}' for v in debug['ref_idf_raw']])}")
            print(f"  IDF-normalized weights gen: {_short_list([f'{v:.4f}' for v in debug['gen_idf_norm']])}")
            print(f"  IDF-normalized weights ref: {_short_list([f'{v:.4f}' for v in debug['ref_idf_norm']])}")
            _print_matrix_preview(debug['sim_matrix'], debug['gen_tokens'], debug['ref_tokens'])
            def _format_vec(vec, head=6, tail=2):
                if len(vec) <= head + tail:
                    return "[" + ", ".join(f"{v:.4f}" for v in vec) + "]"
                head_part = ", ".join(f"{v:.4f}" for v in vec[:head])
                tail_part = ", ".join(f"{v:.4f}" for v in vec[-tail:])
                return f"[{head_part}, ..., {tail_part}]"

            def _print_embedding_preview(tokens, vecs, label, max_rows=4):
                if not vecs:
                    print(f"  {label} embeddings: empty")
                    return
                idxs = list(range(min(max_rows - 1, len(tokens))))
                if len(tokens) > max_rows:
                    idxs.append(len(tokens) - 1)
                print(f"  {label} embeddings (normalized, preview):")
                for i in idxs:
                    tok = tokens[i] if i < len(tokens) else f"t{i}"
                    print(f"    {tok}: {_format_vec(vecs[i])}")
                if len(tokens) > len(idxs):
                    print("    ... (showing first tokens and last token)")

            _print_embedding_preview(debug['gen_tokens'], debug['gen_emb_vecs'], "Generated token")
            _print_embedding_preview(debug['ref_tokens'], debug['ref_emb_vecs'], "Reference token")
            def _print_idf_stats(tokens, df_list, idf_raw, idf_norm, n_docs, label, max_rows=6):
                if not tokens:
                    print(f"  {label} IDF stats: empty")
                    return
                idxs = list(range(min(max_rows - 1, len(tokens))))
                if len(tokens) > max_rows:
                    idxs.append(len(tokens) - 1)
                print(f"  {label} IDF stats (token, df, N, idf, w_norm):")
                for i in idxs:
                    tok = tokens[i] if i < len(tokens) else f"t{i}"
                    df = df_list[i] if i < len(df_list) else 0
                    idf = idf_raw[i] if i < len(idf_raw) else 0.0
                    w = idf_norm[i] if i < len(idf_norm) else 0.0
                    print(f"    {tok}: df={df}, N={n_docs}, idf={idf:.4f}, w={w:.4f}")
                if len(tokens) > len(idxs):
                    print("    ... (showing first tokens and last token)")

            _print_idf_stats(
                debug['gen_tokens'],
                debug['gen_df'],
                debug['gen_idf_raw'],
                debug['gen_idf_norm'],
                debug['idf_N'],
                "Generated token",
            )
            _print_idf_stats(
                debug['ref_tokens'],
                debug['ref_df'],
                debug['ref_idf_raw'],
                debug['ref_idf_norm'],
                debug['idf_N'],
                "Reference token",
            )
            print(f"  Max cosine per gen token (P): {_short_list([f'{v:.4f}' for v in debug['p_max']])}")
            print(f"  Max cosine per ref token (R): {_short_list([f'{v:.4f}' for v in debug['r_max']])}")
            gen_match_preview = []
            for idx, (mx, j) in enumerate(zip(debug['p_max'], debug['p_argmax_idx'])):
                if idx < 8 or idx == len(debug['p_max']) - 1:
                    g_tok = debug['gen_tokens'][idx] if idx < len(debug['gen_tokens']) else f"g{idx}"
                    r_tok = debug['ref_tokens'][j] if 0 <= j < len(debug['ref_tokens']) else f"r{j}"
                    gen_match_preview.append(f"{g_tok}->{r_tok} ({mx:.4f})")
                elif idx == 8:
                    gen_match_preview.append("...")
            print(f"  Example argmax matches (gen->ref): {gen_match_preview}")
            print(f"  P = mean(max cos over gen tokens) = {debug['precision']:.4f}")
            print(f"  R = mean(max cos over ref tokens) = {debug['recall']:.4f}")
            print(f"  F1 = 2 × {debug['precision']:.4f} × {debug['recall']:.4f} / ({debug['precision']:.4f} + {debug['recall']:.4f}) = {debug['f1']:.4f}")
        
        return bert_precision, bert_recall, f1_score

    def context_coverage(self, answer: str, contexts: List[str], verbose: bool = False) -> float:
        ans_doc = self.nlp(answer.lower())
        ans_words = {
            tok.text for tok in ans_doc
            if not tok.is_stop and not tok.is_punct and len(tok.text) > 2
        }
        if not ans_words:
            cov_score = 0.0
        else:
            ctx_doc = self.nlp(" ".join(contexts).lower())
            ctx_words = {tok.text for tok in ctx_doc if not tok.is_punct}
            overlap = len(ans_words & ctx_words)
            cov_score = overlap / len(ans_words)
        
        if verbose:
            print(f"\n[Context Coverage] (Word Overlap)")
            print(f"  Answer: {answer[:100]}...")
            if ans_words:
                ctx_doc = self.nlp(" ".join(contexts).lower())
                ctx_words = {tok.text for tok in ctx_doc if not tok.is_punct}
                overlap_words = len(ans_words & ctx_words)
                def _short_words(items, head=10, tail=10):
                    items = sorted(list(items))
                    if len(items) <= head + tail:
                        return items
                    return items[:head] + ["..."] + items[-tail:]

                print(f"  Content words in answer ({len(ans_words)}): {_short_words(ans_words)}")
                print(f"  Overlap words ({overlap_words}): {_short_words(ans_words & ctx_words)}")
                print(f"  Context Coverage = {overlap_words}/{len(ans_words)} = {cov_score:.4f}")
        
        return cov_score

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        reference: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if reference:
            bert_p, bert_r, bert_f1 = self.bert_f1(reference, answer, verbose=verbose)
            metrics['BERT_PRECISION'] = bert_p
            metrics['BERT_RECALL'] = bert_r
            metrics['BERTSCORE_F1'] = bert_f1
        # metrics['ENTITY_FAITHFULNESS'] = self.entity_faithfulness(answer, contexts)
        metrics['CONTEXT_COVERAGE'] = self.context_coverage(answer, contexts, verbose=verbose)
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
    verbose: bool = False,
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
        idf_sents=ref_df['reference_answer'].dropna().astype(str).tolist(),
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
    for idx, (_, ref_row) in enumerate(ref_df.iterrows()):
        q = ref_row['query']
        ref_ans = ref_row['reference_answer']
        match = gen_df[gen_df['query'] == q]
        if match.empty:
            continue
        gen_ans = match.iloc[0]['generated_answer']
        contexts = match.iloc[0]['contexts']
        if isinstance(contexts, str):
            contexts = _normalize_contexts(contexts)

        show_detail = bool(verbose and idx == 0)

        if show_detail:
            print(f"\n{'='*80}")
            print(f"Query {idx}: {q}")
            print(f"{'='*80}")
        
        # Show detailed step-by-step only for Query 0.
        metrics = evaluator.evaluate(q, gen_ans, contexts, reference=ref_ans, verbose=show_detail)
        metrics.update({
            'query': q,
            'reference_answer': ref_ans,
            'generated_answer': gen_ans,
        })
        results.append(metrics)

    results_df = pd.DataFrame(results)

    # Per-query summary if verbose
    if verbose:
        print(f"\n{'='*80}")
        print("PER-QUERY SUMMARY")
        print(f"{'='*80}")
        for idx, row in results_df.iterrows():
            print(f"\nQuery {idx}: {row['query'][:60]}...")
            if 'BERT_PRECISION' in row and pd.notna(row['BERT_PRECISION']):
                print(f"  BERT Precision:   {row['BERT_PRECISION']:.4f}")
                print(f"  BERT Recall:      {row['BERT_RECALL']:.4f}")
                print(f"  BERTSCORE F1:     {row['BERTSCORE_F1']:.4f}")
            print(f"  Context Coverage: {row['CONTEXT_COVERAGE']:.4f}")

    # Summary stats
    summary = {}
    for col in ['BERT_PRECISION', 'BERT_RECALL', 'BERTSCORE_F1', 'CONTEXT_COVERAGE']:
        if col in results_df.columns:
            summary[col] = {
                'mean': float(results_df[col].mean()),
                'std': float(results_df[col].std()),
                'min': float(results_df[col].min()),
                'max': float(results_df[col].max()),
            }

    results_df.to_csv(output_file, index=False)
    print(f"\n{'='*80}")
    print("SUMMARY - Rata-Rata dari Semua Queries")
    print(f"{'='*80}")
    for k, v in summary.items():
        print(f"  {k}: Mean {v['mean']:.3f} | Std {v['std']:.3f} | Min {v['min']:.3f} | Max {v['max']:.3f}")
    print(f"\nEvaluation complete -> {output_file}")

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NLG evaluation with BERTScore and faithfulness metrics")
    parser.add_argument("--reference", type=str, default="data/query_test.csv")
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
    parser.add_argument("--verbose", action="store_true", help="Show detailed calculations for first query")

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
        verbose=args.verbose,
    )
