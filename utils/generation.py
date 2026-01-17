import ollama
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class GenerationSystem:
    def __init__(self, model_name='gemma2:9b')