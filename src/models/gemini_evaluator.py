import json
import re
import random

from src.config.gemini_config import (
    GOOGLE_API_KEY,
    MODEL_NAME,
    TEMPERATURE,
    MAX_OUTPUT_TOKENS,
    TIMEOUT,
    VALID_LABELS,
    NUM_CONVERSATIONS_TO_PROCESS
)
from src.prompts.prompt import CLASSIFICATION_PROMPT
from src.utils.metrics import display_performance_metrics

def classify_with_gemini(conversation_history, tutor_response):
    # MOCK: For local testing, return random valid labels instead of calling Gemini API
    mi_label = random.choice(VALID_LABELS)
    pg_label = random.choice(VALID_LABELS)
    return mi_label, pg_label

def parse_gemini_output(text):
    # Kept for compatibility with the rest of the codebase
    return classify_with_gemini(None, None)