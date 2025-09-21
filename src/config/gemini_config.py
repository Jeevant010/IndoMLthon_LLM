import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

MODEL_NAME = "gemini-1.5-flash-latest"
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 50
TIMEOUT = 20

VALID_LABELS = ['Yes', 'No', 'To some extent']

NUM_CONVERSATIONS_TO_PROCESS = 1
