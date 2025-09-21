import json
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

# Initialize Gemini API
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in src/config/gemini_config.py")

genai.configure(api_key=GOOGLE_API_KEY)

def parse_gemini_output(text):
    """
    Parse output like:
    Mistake Identification: Yes
    Providing Guidance: To some extent
    """
    mi_label = None
    pg_label = None
    lines = text.strip().split('\n')
    for line in lines:
        if line.startswith("Mistake Identification:"):
            mi_label = line.split(":", 1)[1].strip()
        elif line.startswith("Providing Guidance:"):
            pg_label = line.split(":", 1)[1].strip()
    return mi_label, pg_label

def classify_with_gemini(conversation_history, tutor_response):
    """
    Classifies a tutor's response using the Gemini model.
    Returns a tuple: (mistake_identification_label, providing_guidance_label)
    """
    model = genai.GenerativeModel(MODEL_NAME)

    prompt = CLASSIFICATION_PROMPT.format(
        conversation_history=conversation_history,
        tutor_response=tutor_response,
    )

    try:
        generation_config = genai.types.GenerationConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS
        )

        print(f"  > Waiting for Gemini to classify...")
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            request_options={"timeout": TIMEOUT}  # prevents infinite hang
        )
        output = response.text.strip()
        mi_label, pg_label = parse_gemini_output(output)

        if mi_label not in VALID_LABELS:
            print(f"Warning: Mistake Identification label invalid: '{mi_label}'")
            mi_label = "Error"
        if pg_label not in VALID_LABELS:
            print(f"Warning: Providing Guidance label invalid: '{pg_label}'")
            pg_label = "Error"

        return mi_label, pg_label

    except Exception as e:
        print(f"An error occurred while calling Gemini API: {e}")
        return "Error", "Error"

def run_gemini_evaluation(dataset_path):
    """
    Run evaluation using Gemini model
    """
    print("\n" + "="*60)
    print("RUNNING GEMINI EVALUATION")
    print("="*60)
    
    # Load the dataset
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

    # Performance metrics
    display_performance_metrics(true_labels_mi, predicted_labels_mi, true_labels_pg, predicted_labels_pg, "Gemini")
