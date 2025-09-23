import json
import re
import google.generativeai as genai

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


if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in src/config/gemini_config.py")

genai.configure(api_key=GOOGLE_API_KEY)

def parse_gemini_output(text):
    """
    Robustly parse output like:
    Mistake Identification: Yes
    Providing Guidance: To some extent
    Accepts extra whitespace, punctuation, explanations, and case variations.
    """
    mi_label = None
    pg_label = None
    # Use regex for robustness
    mi_match = re.search(r"mistake identification\\s*:\s*(.+)", text, re.IGNORECASE)
    if mi_match:
        mi_label = mi_match.group(1).strip()
    pg_match = re.search(r"providing guidance\\s*:\s*(.+)", text, re.IGNORECASE)
    if pg_match:
        pg_label = pg_match.group(1).strip()
    return mi_label, pg_label
