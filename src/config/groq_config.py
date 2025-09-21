import os


GROQ_API_KEY = os.getenv("GROQ_API_KEY")


MODEL_NAME = "llama-3.3-70b-versatile"
TEMPERATURE = 0.0
MAX_TOKENS = 30


VALID_LABELS = ['Yes', 'No', 'To some extent']


NUM_CONVERSATIONS_TO_PROCESS = 1
