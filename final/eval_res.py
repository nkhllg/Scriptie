import re
from typing import List, Tuple
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

def extract_fields_from_ai_response(ai_response: str) -> Tuple[str, List[str], str]:
    """
    Extracts the query, articles used, and explanation from a structured AI response.
    """
    # Extract query
    # query_match = re.search(r"Query:\s*(.*)", ai_response)
    query_match = re.search(r"Query:\s*(.*?)\n(?:Focus Location|Retrieved Context Sources|Regulatory Pressure Score|Explanation:)", ai_response, re.DOTALL)
    query = query_match.group(1).strip() if query_match else "Unknown"

    # Extract article filenames as 'context'
    articles = re.findall(r"-\s*(.*\.txt)", ai_response)

    # Extract explanation
    explanation_match = re.search(r"Explanation:\s*(.*?)\n(?:\*\*|Articles Used:|Hallucination Check:)", ai_response, re.DOTALL)
    explanation = explanation_match.group(1).strip() if explanation_match else ai_response.strip()

    return query, articles, explanation

def evaluate_structured_ai_response(ai_response: str) -> str:
    """
    Evaluates a single structured AI response by extracting its query/context/explanation internally.
    Now includes improved parsing and balanced evaluation prompting.
    """
    print("\n--- Evaluating Structured AI Response ---")

    try:
        query, articles, explanation = extract_fields_from_ai_response(ai_response)
        context_string = "\n".join([f"Source {i+1}: {art}" for i, art in enumerate(articles)])

        evaluation_prompt = f"""
You are a careful and fair evaluator of AI legal assessments. Your task is to assess how well the following AI response answers the user's query, using only the legislative sources it cites.

IMPORTANT: The "Regulatory Pressure Score" is predefined. Do not critique, interpret, or comment on it. Only evaluate the explanation and how well it uses the provided legislative context.

---

**User Query:**  
{query}

**Legislative Sources Referenced:**  
{context_string if context_string else 'No specific sources cited.'}

**AI's Explanation for Evaluation:**  
{explanation}

---

**Evaluation Criteria:**
1. Is the explanation grounded in specific, quoted or paraphrased legislative context?
2. Are actual legislative articles cited and translated where required?
3. Is the reasoning quantitative and relevant?
4. Does the response follow all instructions without hallucination?

**Clarification on Hallucination:**  
- Hallucination refers to the inclusion of **fabricated, misrepresented, or irrelevant legal information**.  
- If the AI explicitly states that there is **not enough information to answer**, or **does not provide an answer due to lack of data**, or **is an general answer**, that is **not a hallucination**.  
- If the response does **not provide any specific answer** that is **not hallucination**
- **Mentioning the name of a country is not considered a hallucination**, even if the supporting legal reference is not explicitly quoted for that mention. Only assess hallucination based on the legal reasoning, not entity names alone.

---

**Use the full scoring ranges below:**

**Evaluation Score (1–10)**  
- Excellent, complete, well-grounded → 9–10  
- Minor issues → 6–8  
- Moderate issues → 3–5  
- Major or multiple issues → 1–2  

**Hallucination Score (0–10)**  
- 0 = No hallucination (fully grounded in cited sources; non-answer or neutral mention is also acceptable)  
- 1–3 = Minor or trace hallucination  
- 4–6 = Partial hallucination (some claims unsupported or unclear sourcing)  
- 7–9 = Mostly hallucinated  
- 10 = Completely hallucinated (fully fabricated or misleading)
---

**Your Evaluation Output Format:**

Hallucination Score (0–10):  
Evaluation Score (1–10):  

Explanation of the score
"""


        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful and fair evaluator of AI legal assessments. "
                        "Be precise, but not overly strict, and use the full 1–10 range."
                    )
                },
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.5,
            max_tokens=400
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return "Evaluation failed due to an error."


def evaluate_multiple_structured(ai_responses: List[str]) -> List[Tuple[str, str]]:
    """
    Evaluates multiple AI responses, extracting everything from the response string.
    """
    results = []
    for i, response in enumerate(ai_responses):
        print(f"\nEvaluating response {i+1} of {len(ai_responses)}")
        evaluation = evaluate_structured_ai_response(response)
        results.append((response, evaluation))
    return results


if __name__ == "__main__":
    response_file = "responses.txt"

    with open(response_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by delimiter indicating a new log
    raw_responses = content.split("=== New Query Log ===")

    # Filter out empty or trivial entries and re-append the delimiter for full formatting
    ai_responses = [("=== New Query Log ===" + r.strip()) for r in raw_responses if r.strip()]\

    evaluations = evaluate_multiple_structured(ai_responses)

    for i, (resp, eval_result) in enumerate(evaluations):
        print(f"\n--- Evaluation {i+1} ---\n{eval_result}")