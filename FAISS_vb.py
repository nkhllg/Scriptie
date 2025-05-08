import os
import re
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import openai
from dotenv import load_dotenv
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
load_dotenv()

DATA_FOLDER = "legislation_data"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
TEXT_CHUNK_SIZE = 1000
TEXT_CHUNK_OVERLAP = 150

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
vector_size = embedding_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(vector_size)
metadata_store = {}

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=TEXT_CHUNK_SIZE,
    chunk_overlap=TEXT_CHUNK_OVERLAP,
    length_function=len,
)

def parse_filename(filename: str) -> Optional[Dict[str, str]]:
    pattern = re.compile(r"^(?P<location>.+?)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<desc>.*)\.txt$", re.IGNORECASE)
    match = pattern.match(filename)
    if match:
        data = match.groupdict()
        try:
            datetime.strptime(data['date'], '%Y-%m-%d')
            return {
                "location": data['location'].replace('_', ' ').strip(),
                "date_iso": data['date'],
                "description": data['desc'].replace('_', ' ').strip()
            }
        except ValueError:
            return {
                "location": data['location'].replace('_', ' ').strip(),
                "date_iso": "unknown",
                "description": data['desc'].replace('_', ' ').strip()
            }
    return None

def extract_metadata_from_path(file_path: str, text: str) -> Dict[str, str]:
    location = os.path.basename(os.path.dirname(file_path)) or "Unknown"
    date_match = re.search(r"(?:operation on|amendments up to and including)\s+(\d{1,2} \w+ \d{4})", text)
    try:
        date_parsed = datetime.strptime(date_match.group(1), "%d %B %Y").date().isoformat() if date_match else "unknown"
    except ValueError:
        date_parsed = "unknown"
    title_match = re.search(r"(BUILDING CONTROL ACT \d{4}(?:.*?)?)\n", text, re.IGNORECASE)
    description = title_match.group(1).strip() if title_match else os.path.basename(file_path)
    return {
        "location": location,
        "date_iso": date_parsed,
        "description": description
    }

def index_data_from_folder(folder_path: str, target_location: str):
    print(f"\n--- Indexing data from: {folder_path} ---")
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return

    for root, _, files in os.walk(folder_path):
        if os.path.basename(root) == target_location:
            print(f"Found target folder: {root}")
            for filename in files:
                if filename.lower().endswith(".txt"):
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                    except Exception as e:
                        print(f"Error reading file {filename}: {e}. Skipping.")
                        continue

                    if not text.strip():
                        print(f"Warning: File {filename} is empty. Skipping.")
                        continue

                    metadata = parse_filename(filename) or extract_metadata_from_path(file_path, text)
                    chunks = text_splitter.split_text(text)
                    embeddings = embedding_model.encode(chunks)

                    for i, chunk in enumerate(chunks):
                        index.add(np.array([embeddings[i]], dtype=np.float32))
                        metadata_store[index.ntotal - 1] = {
                            "text": chunk,
                            "original_filename": filename,
                            "chunk_index": i,
                            **metadata
                        }


def retrieve_relevant_context(query: str, top_k: int = 5, location_filter: Optional[str] = None) -> List[Dict]:
    print(f"\n--- Retrieving context for query: '{query}' ---")
    query_vector = embedding_model.encode([query]).astype(np.float32)
    D, I = index.search(query_vector, top_k)

    context = []
    for idx in I[0]:
        if idx == -1 or idx not in metadata_store:
            continue
        item = metadata_store[idx]
        if location_filter and item.get("location") not in [location_filter, "National", "Unknown"]:
            continue
        print(f"Found: Location = {item['location']}, File = {item['original_filename']}")
        context.append(item)
    return context

def get_regulatory_assessment(query: str, context: List[Dict]) -> Tuple[Optional[int], Optional[str]]:
    print("\n--- Generating assessment with OpenAI ---")
    if not context:
        return None, "No relevant context found in the database to assess the query."

    context_string = ""
    for i, item in enumerate(context):
        source_info = f"Source {i+1}: [Location: {item['location']}, File: {item['original_filename']}]"
        context_string += f"{source_info}\nContent: {item['text']}\n\n"

    prompt = f"""
You are an AI assistant specialized in analyzing real estate regulations. Your task is to assess the regulatory pressure based *only* on the provided context below.

**User Query:** {query}

**Context from Legislative Documents:**
--- START CONTEXT ---
{context_string}
--- END CONTEXT ---

**Instructions:**
1.  Carefully review the provided context in relation to the user query.
2.  Assess the level of regulatory pressure concerning "{query}". Consider factors like restrictions, landlord/developer obligations, tenant protections, complexity of compliance, etc. shown in the text.
3.  Provide a **Regulatory Pressure Score** on a scale of 1 to 10, where:
    *   1 means Very Low Pressure (e.g., very permissive, few restrictions, highly favorable to developers/landlords).
    *   5 means Moderate Pressure (e.g., balanced regulations, standard compliance).
    *   10 means Very High Pressure (e.g., very restrictive, strong tenant protections, complex and burdensome for developers/landlords).
4.  Provide a concise **Explanation** for your score. Justify your reasoning by referencing specific information or quotes from the provided context (referencing "Source 1", "Source 2", etc. is helpful).
5.  **IMPORTANT:** Base your entire assessment *strictly* on the provided text context. Do not use any external knowledge. If the context is insufficient or irrelevant to the query, state that clearly and assign a score of N/A.

**Output Format:**
Score: [Your Score from 1-10 or N/A]
Explanation: [Your explanation, referencing the context]
"""

    try:
        print("Sending request to OpenAI...")
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo", # Or "gpt-4"
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in analyzing real estate regulations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        ai_message = response.choices[0].message.content.strip()
        print("OpenAI Assessment Received.")

        # --- Parse the response ---
        score = None
        explanation = ai_message # Default

        score_match = re.search(r"Score:\s*(\d+|N/A)", ai_message, re.IGNORECASE)
        explanation_match = re.search(r"Explanation:\s*(.*)", ai_message, re.DOTALL | re.IGNORECASE)

        if score_match:
            score_str = score_match.group(1)
            if score_str.upper() == "N/A":
                score = None
                explanation = "N/A - Context likely insufficient or irrelevant as assessed by AI."
                print("Score: N/A")
            else:
                try:
                    score = int(score_str)
                    print(f"Score: {score}")
                except ValueError:
                    print(f"Warning: Could not parse score '{score_str}' as integer.")
                    score = None # Treat invalid number as N/A

        if explanation_match:
            explanation = explanation_match.group(1).strip()
            print(f"Explanation: {explanation}")
        else:
             print("Warning: Could not parse explanation separately. Returning full AI response as explanation.")

        return score, explanation

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None, f"An error occurred during AI assessment: {e}"

# --- Main Execution ---
if __name__ == "__main__":
    query = "Restrictions on maximum rent increase"
    target_location = "Singapore"

    index_data_from_folder(DATA_FOLDER, target_location)

    print("\n" + "="*50)
    print("Example Usage: Assessing Regulatory Pressure")
    print("="*50)

    relevant_context = retrieve_relevant_context(query, top_k=5, location_filter=target_location)
    if relevant_context:
        score, explanation = get_regulatory_assessment(query, relevant_context)
        print(f"\nQuery: {query}")
        print(f"Score: {score if score is not None else 'N/A'}")
        print(f"Explanation: {explanation}")
    else:
        print("\nNo relevant documents found for this location.")


