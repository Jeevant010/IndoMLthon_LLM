HUGGINGFACE_PROMPT = """
You are an evaluator. Your job is to assess whether the tutor’s response in the conversation:
1. Correctly identifies a mistake in the student’s work.
2. Provides useful guidance to the student.

Conversation:
{conversation_history}

Tutor Response:
{tutor_response_text}

Your task:
Return ONLY the two labels in the following format (no extra words, no explanation):

Mistake identification: <Yes/No/To some extent>
Provided guidance: <Yes/No/To some extent>
"""
