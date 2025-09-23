# -------------------------------
# FEW-SHOT PROMPT TEMPLATE for Gemini Classification
# -------------------------------

CLASSIFICATION_PROMPT = """
You are evaluating a tutor's reply to a student's question. Your task is to make **two separate judgments** based on the conversation history and tutor's response.

First, carefully read the conversation history and the tutor's reply.
Then, think step-by-step about whether the tutor identified the student's mistake, and whether they provided helpful guidance.

For each judgment, choose exactly one of these options:
- Yes: Fully and clearly identified the mistake or provided helpful guidance.
- No: Did not identify the mistake or failed to provide meaningful guidance.
- To some extent: Partially identified the mistake or gave incomplete or somewhat helpful guidance.

Return your answers in the following exact format (without any extra text or punctuation):
Mistake Identification: <Yes/No/To some extent>
Providing Guidance: <Yes/No/To some extent>

---

## FEW-SHOT EXAMPLES

### Example 1
Conversation History:
---
Tutor: You earn one point for your good beginning.
Tutor: That was a good try.
Tutor: What is the value of 3^3?
Student: 9
---
Tutor's Response:
"Remember, 3 to the power of 3 means 3 multiplied by itself three times: 3 x 3 x 3."
Mistake Identification: Yes
Providing Guidance: Yes

### Example 2
Conversation History:
---
Tutor: Hi, could you please provide a step-by-step solution for the question below? The question is: Tyson decided to make muffaletta sandwiches for the big game. ...
---
Tutor's Response:
"You're absolutely right about the cost of meat, but remember that each sandwich needs both meat and cheese, so we need to figure out the total cost of both ingredients for each."
Mistake Identification: Yes
Providing Guidance: To some extent

### Example 3
Conversation History:
---
Student: okey
Tutor: Now we have the same denominators so we can subtract the numerators directly.
Tutor: What is 25 minus 18?
Student: 8
---
Tutor's Response:
"That's great! So our answer is 8/20.  Remember, we can always try to simplify our fractions if possible!"
Mistake Identification: No
Providing Guidance: No

### Example 4 (New edge case)
Conversation History:
---
Tutor: The product of 7 and 4 is 28.
Student: 25.
---
Tutor's Response:
"Actually, 7 times 4 is 28, not 25. Let's review multiplication tables."
Mistake Identification: Yes
Providing Guidance: Yes

---

### Now, classify the following:

Conversation History:
---
{conversation_history}
---

Tutor's Response:
"{tutor_response}"

Think step-by-step, then output your answers in the format:
Mistake Identification: <Yes/No/To some extent>
Providing Guidance: <Yes/No/To some extent>
"""