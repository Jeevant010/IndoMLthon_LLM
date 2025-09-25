# Gemini model configuration for tutor response classification

import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

MODEL_NAME = "gemini-1.5-flash-latest"


TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 128
TIMEOUT = 30  # in seconds

# Allowed output labels for robustness
VALID_LABELS = ["Yes", "No", "To some extent", "Error"]

# Number of conversations to process in evaluation (can be tuned)
NUM_CONVERSATIONS_TO_PROCESS = 50