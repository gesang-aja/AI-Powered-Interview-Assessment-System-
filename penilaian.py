question1 = """Can you share any specific challenges you faced while working on certification and how you overcame them?"""
rubric1 = """
Score 4:
- Comprehensive and Clear Response
- Provides a detailed description of specific challenges encountered during the certification.
- Offers clear explanations of how each challenge was overcome.
- Demonstrates strong understanding of technical aspects and problem-solving skills.
- Response is very clear, well-organized, and reflects on the learning process.

Score 3:
- Specific Challenge with Basic Solution
- Describes at least one specific challenge related to building machine learning models with TensorFlow.
- Provides a basic explanation of how the challenge was overcome
- Explanation may be brief or lack depth
- Shows some understanding but lacks detailed insight

Score 2:
- General response with limited details

Score 1:
- Minimal or vague response

Score 0:
- Unanswered or irrelevant
"""

question2 = """Can you describe your experience with transfer learning in TensorFlow? How did it benefit your projects?"""
rubric2 = """
Score 4:
- Comprehensive and Very Clear Response.
- Offers a detailed description of personal experience using transfer learning in TensorFlow.
- Provides specific examples of projects and clearly explains the benefits gained.
- Demonstrates strong understanding of transfer learning concepts and their practical application.
- Response is very clear, well-organized, and reflects deep engagement with the subject.


Score 3:
- Specific Experience with Basic Explanation.
- Describes personal experience with transfer learning in TensorFlow.
- Provides examples of projects where transfer learning was applied.
- Explains how transfer learning benefited those projects.
- Explanation may be brief or lack comprehensive insight.

Score 2:
- General response with limited details
- Mentions transfer learning or TensorFlow in general terms.
- Provides minimal details about personal experience.
- Does not clearly explain how transfer learning benefited projects.
- Shows basic understanding but lacks depth and specificity.

Score 1:
- Minimal or vague response

Score 0:
- Unanswered or irrelevant
"""

question3 = """Describe a complex TensorFlow model you have built and the steps you took to ensure its accuracy and efficiency."""

rubric3 = """"
Score 4:
- Comprehensive and Very Clear Response.
- Offers a detailed description of a complex TensorFlow model built.
- Provides specific details about the model's architecture, features, and purpose.
- Clearly explains the steps taken to ensure both accuracy and efficiency, such as data preprocessing, model optimization techniques, regularization methods, and performance tuning.
- Demonstrates strong understanding of TensorFlow and machine learning concepts.
- Response is very clear, well-organized, and reflects deep engagement with the subject.

Score 3:
- Specific Model with Basic Explanation.
- Describes a complex TensorFlow model they have built.
- Provides some details about the model's architecture or purpose.
- Explains steps taken to ensure accuracy and efficiency, but may lack depth.
- Explanation may be brief or not fully comprehensive.

Score 2:
- General Response with Limited Details
- Mentions building a TensorFlow model in general terms.
- Provides minimal details about the model's complexity.
- Does not clearly explain the steps taken to ensure accuracy and efficiency.
- Shows basic understanding but lacks depth and specificity.

Score 1: 
- Minimal or Vague Response

Score 0:
- Unanswered
"""


question4 = """Explain how to implement dropout in a TensorFlow model and the effect it has on training."""

rubric4 = """
Score 4:
- Comprehensive and Very Clear Response.
- Provides a detailed explanation of how to implement dropout in a TensorFlow model, including code examples or specific functions (e.g., using tf.keras.layers.Dropout).
- Clearly explains the effect of dropout on training, such as how it helps prevent overfitting by randomly deactivating neurons during training.
- Discusses the impact on model performance, generalization, and possibly mentions considerations like dropout rates.
- Demonstrates strong understanding of TensorFlow and machine learning concepts.
- Response is very clear, well-organized, and reflects deep engagement with the subject.

Score 3:
- Specific Explanation with Basic Understanding
- Explains how to implement dropout in a TensorFlow model with some specifics (e.g., mentions using Dropout layer).
- Describes the general effect of dropout on training, such as preventing overfitting.
- May omit some details about implementation or effects.
- Demonstrates a reasonable understanding but lacks comprehensive insight.

Score 2:
- General Response with Limited Details

Score 1:
- Minimal or Vague Response

Score 0:
- Unanswered
"""

question5 = """Describe the process of building a convolutional neural network (CNN) using TensorFlow for image classification."""

rubric5 = """
Score 4:
- Comprehensive and Very Clear Response
- Provides a detailed, step-by-step description of building a CNN in TensorFlow for image classification.
- Covers all key components, including data loading and preprocessing, defining the CNN layers (convolutional, pooling, activation functions), compiling the model with appropriate loss function and optimizer, training the model, and evaluating performance.
- May include code examples or specific TensorFlow functions used.
- Demonstrates strong understanding of CNNs and TensorFlow.
- Response is very clear, well-organized, and reflects deep engagement with the subject.

Score 3:
- Specific Explanation with Basic Understanding
- Describes the process of building a CNN in TensorFlow with some specifics.
- Includes key steps such as data preprocessing, defining the CNN architecture, compiling the model, and training
- May lack comprehensive detail or omit some important aspects.
- Demonstrates reasonable understanding but may not fully elaborate on each step.

Score 2:
- General Response with Limited Details

Score 1: 
- Minimal or Vague Response

Score 0:
- Unanswered
"""




import json
from google.api_core.exceptions import ResourceExhausted, GoogleAPIError

# pastikan ini sudah dipanggil sekali di awal program
# genai.configure(api_key="API_KEY_LO")
# model = genai.GenerativeModel("gemini-2.5-flash")


def penilaian_interview(question, rubrik, answer, model_ai):
    prompt = f"""
You are an expert technical interviewer evaluating an answer.

QUESTION:
{question}

RUBRIC:
{rubrik}

ANSWER:
\"\"\"
{answer}
\"\"\"

TASK:
- Assign a score from 0 to 4 strictly based on the rubric
- Do NOT infer information that is not explicitly stated
- If the answer is irrelevant or does not address the question, return score 0
- Provide a brief justification referencing the rubric
- Return VALID JSON ONLY

JSON FORMAT:
{{
  "score": <integer 0-4>,
  "justification": "<short explanation>"
}}
"""

    try:
        response = model_ai.generate_content(prompt)
        raw_text = response.text.strip()

        # Extract JSON safely
        json_text = raw_text[
            raw_text.find("{") : raw_text.rfind("}") + 1
        ]

        result = json.loads(json_text)

        # Safety validation
        if "score" not in result or "justification" not in result:
            raise ValueError("Invalid JSON structure")

        return result

    except ResourceExhausted:
        return {
            "score": 0,
            "justification": "AI quota exceeded. Manual review required."
        }

    except (GoogleAPIError, json.JSONDecodeError, ValueError):
        return {
            "score": 0,
            "justification": "AI evaluation failed. Manual review required."
        }

    except Exception:
        return {
            "score": 0,
            "justification": "Unexpected error during AI evaluation."
        }