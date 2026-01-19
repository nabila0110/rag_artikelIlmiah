import ollama
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class GenerationSystem:
    def __init__(self, model_name='gemma2:9b', temperature=0.3, max_tokens=500):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self._test_connection()

        logger.info(f"Generation system initialized with model: {model_name}")

    def _test_connection(self):
        try:
            ollama.list()
            logger.info("Ollama connection success")
        except Exception as e:
            logger.error(f"cannot connect to ollama: {e}")
            raise ConnectionError("Ollama service not available. Please start ollama")
        
    def generate_answer(self, query: str, retrieved_chunks: List[Dict], max_context_chunks: int = 5) -> Dict:
        try:
            context = self._build_context(retrieved_chunks[:max_context_chunks])
            prompt = self._build_prompt(query, context)

            response = ollama.generate(
                model=self.model_name,
                prompt = prompt,
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens
                }
            )

            answer = response['response']
            logger.info(f"Generated answer for query: '{query}")

            return{
                'answer': answer,
                'context_chunks_used': len(retrieved_chunks[:max_context_chunks]),
                'model': self.model_name
            }
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return{
                'answer': f"Maaf, terjadi kesalahan dalam menghasilkan jawaban: {str(e)}",
                'context_chunks_used': 0,
                'model': self.model_name,
                'error': str(e)
            }
        
    def _build_context(self, chunks: List[Dict])->str:
        context_parts=[]

        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"\n[Sumber {i}] {chunk['judul']} ({chunk['tahun']})\n"
                f"Penulis: {chunk['author']}\n"
                f"Section: {chunk['section']}\n"
                f"{chunk['chunk_text']}\n"
            )

        return "\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str)->str:
        prompt = f"""Anda adalah asisten penelitian. Jawab pertanyaan berdasarkan sumber yang diberikan.

    Pertanyaan: {query}

    Sumber-sumber:
    {context}

    instruksi
    1. Jawab secara komprehensif dalam Bahasa Indonesia formal
    2. WAJIB gunakan citation [1], [2] dll diakhir setiap klaim
    3. Hanya gunakan informasi dari sumber
    4. Jika tidak cukup info, katakan "informasi tidak tersedia"
    5. Tulis dalam bentuk paragraf yang koheren dan mudah dipahami

    Jawaban:"""
        
        return prompt
