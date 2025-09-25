import json
import os
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

genai.configure(api_key=GOOGLE_API_KEY)

def parse_gemini_output(text):
    """
    Parses Gemini's output and returns (Mistake_Identification, Providing_Guidance)
    """
    mistake = None
    guidance = None
    for line in text.splitlines():
        if line.lower().startswith("mistake identification:"):
            mistake = line.split(":")[1].strip()
        elif line.lower().startswith("providing guidance:"):
            guidance = line.split(":")[1].strip()
    return mistake, guidance

def classify_with_gemini(conversation_history, tutor_response):
    prompt = CLASSIFICATION_PROMPT.format(
        conversation_history=conversation_history,
        tutor_response=tutor_response
    )
    response = genai.generate_text(
        model=MODEL_NAME,
        prompt=prompt,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        stop_sequences=None,
        timeout=TIMEOUT
    )
    text = response.result
    mi_label, pg_label = parse_gemini_output(text)
    # fallback to "Error" if parsing fails
    if mi_label not in VALID_LABELS:
        mi_label = "Error"
    if pg_label not in VALID_LABELS:
        pg_label = "Error"
    return mi_label, pg_label

def run_gemini_evaluation(dataset_path):
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
    prediction_data = []

    for conversation in dev_data[:NUM_CONVERSATIONS_TO_PROCESS]:
        conversation_id = conversation['conversation_id']
        conversation_history = conversation['conversation_history']
        tutor_responses = conversation['tutor_responses']

        print(f"\n--- Conversation ID: {conversation_id} ---")
        conv_obj = {
            "conversation_id": conversation_id,
            "conversation_history": conversation_history,
            "tutor_responses": {}
        }

        for tutor_name, response_data in tutor_responses.items():
            tutor_response_text = response_data['response']
            true_mi = response_data['annotation'].get('Mistake_Identification', None)
            true_pg = response_data['annotation'].get('Providing_Guidance', None)
            true_labels_mi.append(true_mi)
            true_labels_pg.append(true_pg)

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
            conv_obj["tutor_responses"][tutor_name] = {
                "response": tutor_response_text,
                "annotation": {
                    "Mistake_Identification": pred_mi,
                    "Providing_Guidance": pred_pg
                }
            }
        prediction_data.append(conv_obj)

    display_performance_metrics(true_labels_mi, predicted_labels_mi, true_labels_pg, predicted_labels_pg, "Gemini")
    output_path = os.path.join(os.path.dirname(dataset_path), "prediction.json")
    with open(output_path, "w") as f:
        json.dump(prediction_data, f, indent=2)
    print(f"Predictions written to {output_path}")
