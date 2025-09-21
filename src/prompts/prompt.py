# -------------------------------
# UNIFIED PROMPT TEMPLATE
# -------------------------------

CLASSIFICATION_PROMPT = """
You are evaluating a tutor's reply to a student's question. Your task is to make **two separate judgments** based on the conversation history and tutor's response:

1. Mistake Identification: Did the tutor correctly identify the student's mistake?  
2. Providing Guidance: Did the tutor provide proper guidance to help the student?

For each judgment, choose exactly one of these options:  
- Yes: Fully and clearly identified the mistake or provided helpful guidance.  
- No: Did not identify the mistake or failed to provide meaningful guidance.  
- To some extent: Partially identified the mistake or gave incomplete or somewhat helpful guidance.

Return your answers in the following exact format (without any extra text or punctuation):  
Mistake Identification: <Yes/No/To some extent>  
Providing Guidance: <Yes/No/To some extent>

---

### Input for Classification

Conversation History:  
---  
{conversation_history}  
---  

Tutor's Response:  
"{tutor_response}"
"""
