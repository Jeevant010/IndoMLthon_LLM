# LLM Baseline Model Evaluation

Automated evaluation of LLM models for tutor response classification using Gemini, Groq APIs and HuggingFace.

## Run on colab

### Install dependencies
```bash
# Clone the github repo
!git clone https://github.com/JayeshAgarwal03/LLM_baseLineModel.git
# Move into the repo
%cd LLM_baseLineModel/
# Install dependencies
!pip install -r requirements.txt

# Set up environment variables
%cp env.example .env
# Edit .env with your actual API keys

# Run evaluation
!python main.py --model gemini --dataset path_to_dataset
!python main.py --model groq --dataset path_to_dataset
!python main.py --model huggingface --dataset path_to_dataset
```

**Arguments:**
- `--model, -m`: Model to evaluate (`gemini`, `groq`, or `huggingface`)
- `--dataset, -d`: Path to dataset JSON file

## Configuration

### API Models (Gemini/Groq) local hugging_face Models(google/gemma-2-2b-it or mistralai/Mistral-7B-Instruct-v0.3 or meta-llama/Llama-3.3-70B-Instruct)
1. Copy `env.example` to `.env`
2. Set your API keys and hugging_face tokens in the `.env` file:
   ```
   GOOGLE_API_KEY=your_actual_gemini_key
   GROQ_API_KEY=your_actual_groq_key
   HF_TOKEN=your_actual_hf_token
   ```
3. The `.env` file is automatically ignored by git for security


### Hugging Face Model
The Hugging Face model uses `google/gemma-2-2b-it` by default. You can configure it in `src/config/hugging_face_config.py`:
- `HF_TOKEN`: Go to this link: https://huggingface.co/settings/tokens and generate your own hugging face token. You'll have to agree to the terms and conditions. After accepting it, set the token in the `.env` file or directly in the config if not using an environment variable.
- `MODEL_NAME`: Change the model (default: "google/gemma-2-2b-it", also supports "mistralai/Mistral-7B-Instruct-v0.3" and "meta-llama/Llama-3.3-70B-Instruct")
- `MAX_CONVERSATIONS`: Adjust the maximum number of conversation turns to keep in memory (default: 5)
- `MAX_NEW_TOKENS`: Set the maximum number of new tokens to generate in the model's response (default: 100)

## Prompts

You can edit `prompt.py`