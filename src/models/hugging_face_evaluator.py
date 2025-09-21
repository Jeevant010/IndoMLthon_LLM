import os
import re
import json
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config.hugging_face_config import MODEL_NAME, HF_TOKEN, MAX_CONVERSATIONS, MAX_NEW_TOKENS
from src.prompts.huggingFace_prompts import HUGGINGFACE_PROMPT
from src.utils.metrics import display_performance_metrics

load_dotenv()

class HuggingFaceEvaluator:
    def __init__(self, model_name=MODEL_NAME):
        token = HF_TOKEN
        if not token:
            raise ValueError("HF_TOKEN environment variable not set.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  
            device_map="auto",
            token=token
        )
        self.model.eval()


    def evaluate(self, conversation_history, tutor_response_text):
        prompt = HUGGINGFACE_PROMPT.format(
            conversation_history=conversation_history,
            tutor_response_text=tutor_response_text
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        mistake = re.search(r"Mistake identification:\s*(Yes|No|To some extent)", response, re.IGNORECASE)
        guidance = re.search(r"Provided guidance:\s*(Yes|No|To some extent)", response, re.IGNORECASE)

        pred_mistake = mistake.group(1) if mistake else "No"
        pred_guidance = guidance.group(1) if guidance else "No"

        return {
            "pred_mistake": pred_mistake,
            "pred_guidance": pred_guidance
        }

def run_huggingface_evaluation(dataset_path):
    print("Running Hugging Face evaluation...")
    try:
        with open(dataset_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
        return
    
    evaluator = HuggingFaceEvaluator()
    true_labels_mi, predicted_labels_mi = [], []
    true_labels_pg, predicted_labels_pg = [], []

    for conversation in data[:MAX_CONVERSATIONS]:
        conversation_id = conversation["conversation_id"]
        conversation_history = conversation["conversation_history"]
        tutor_responses = conversation["tutor_responses"]

        print(f"\n--- Conversation ID: {conversation_id} ---")

        for tutor_name, response_data in tutor_responses.items():
            tutor_response_text = response_data["response"]
            true_mistake = response_data["annotation"]["Mistake_Identification"]
            true_guidance = response_data["annotation"]["Providing_Guidance"]

            result = evaluator.evaluate(conversation_history, tutor_response_text)
            pred_mistake = result["pred_mistake"]
            pred_guidance = result["pred_guidance"]

            true_labels_mi.append(true_mistake)
            true_labels_pg.append(true_guidance)
            predicted_labels_mi.append(pred_mistake)
            predicted_labels_pg.append(pred_guidance)

            print(f"Tutor: {tutor_name}")
            print(f"Response: {tutor_response_text}")
            print(f"Mistake ID -> True: {true_mistake}, Predicted: {pred_mistake}")
            print(f"Guidance   -> True: {true_guidance}, Predicted: {pred_guidance}")
            print("-" * 20)

    display_performance_metrics(true_labels_mi, predicted_labels_mi, true_labels_pg, predicted_labels_pg, "Hugging Face")


