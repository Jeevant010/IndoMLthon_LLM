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

def parse_gemini_output(text):
    # Kept for compatibility, but not used with the mock
    return classify_with_gemini(None, None)

def classify_with_gemini(conversation_history, tutor_response):
    # MOCK: For local testing, return random valid labels instead of calling Gemini API
    mi_label = random.choice(VALID_LABELS)
    pg_label = random.choice(VALID_LABELS)
    return mi_label, pg_label

def run_gemini_evaluation(dataset_path):
    """
    Run evaluation using Gemini model
    """
    print("\n" + "="*60)
    print("RUNNING GEMINI EVALUATION")
    print("="*60)

    try:
        with open(dataset_path, 'r') as f:
            dev_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_path}")
        return

    true_labels_mi, predicted_labels_mi = [], []
    true_labels_pg, predicted_labels_pg = [], []

    for conversation in dev_data[:NUM_CONVERSATIONS_TO_PROCESS]:
        conversation_id = conversation['conversation_id']
        conversation_history = conversation['conversation_history']
        tutor_responses = conversation['tutor_responses']

        print(f"\n--- Conversation ID: {conversation_id} ---")

        for tutor_name, response_data in tutor_responses.items():
            tutor_response_text = response_data['response']

            # Ground truth
            true_mi = response_data['annotation']['Mistake_Identification']
            true_pg = response_data['annotation']['Providing_Guidance']

            true_labels_mi.append(true_mi)
            true_labels_pg.append(true_pg)

            # Single prediction returning both labels
            pred_mi, pred_pg = classify_with_gemini(
                conversation_history,
                tutor_response_text
            )
            predicted_labels_mi.append(pred_mi)
            predicted_labels_pg.append(pred_pg)

            print(f"Tutor: {tutor_name}")
            print(f"Response: {tutor_response_text}")
            print(f"Mistake ID -> True: {true_mi}, Predicted: {pred_mi}")
            print(f"Guidance   -> True: {true_pg}, Predicted: {pred_pg}")
            print("-" * 20)

    display_performance_metrics(true_labels_mi, predicted_labels_mi, true_labels_pg, predicted_labels_pg, "Gemini")
