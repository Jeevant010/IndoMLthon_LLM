import os

HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_NAME = "google/gemma-2-2b-it"
#following models can also be used:
#mistralai/Mistral-7B-Instruct-v0.3
#meta-llama/Llama-3.3-70B-Instruct
MAX_CONVERSATIONS = 5
MAX_NEW_TOKENS = 100
