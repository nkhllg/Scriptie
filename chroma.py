import os
import re
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
from geolocation import get_location_nominatim

address = "Oude Delft 169"
target_location = get_location_nominatim(address)
print("target location found: ", target_location.get('country'))

# --- Configuration ---
load_dotenv()

DATA_FOLDER = os.path.join("..", "Data_txts", target_location.get('country', '').strip())
COLLECTION_NAME = "real_estate_regulations_temp" # Use a distinct name maybe
# EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
# EMBEDDING_MODEL = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
# EMBEDDING_MODEL = 'distiluse-base-multilingual-cased-v2'
EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
# EMBEDDING_MODEL = 'sentence-transformers/LaBSE'
TEXT_CHUNK_SIZE = 1000  # Characters per chunk
TEXT_CHUNK_OVERLAP = 150 # Overlap between chunks

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

FILENAME_PATTERN = re.compile(r"^(?P<location>.+?)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<desc>.*)\.txt$", re.IGNORECASE)

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
vector_size = embedding_model.get_sentence_embedding_dimension()

chroma_client = chromadb.Client(Settings())
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=TEXT_CHUNK_SIZE,
    chunk_overlap=TEXT_CHUNK_OVERLAP,
    length_function=len,
)

def parse_filename(filename: str) -> Optional[Dict[str, str]]:
    match = FILENAME_PATTERN.match(filename)
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
                "date_iso": None,
                "description": data['desc'].replace('_', ' ').strip()
            }
    return None

def extract_metadata_from_path(file_path: str, text: str) -> Dict[str, str]:
    """Extract metadata using the file path (for location) and content (for optional date/description)."""
    # Get parent folder as location
    location = os.path.basename(os.path.dirname(file_path)) or "Unknown"

    # Optional: Extract date from text
    date_match = re.search(r"(?:operation on|amendments up to and including)\s+(\d{1,2} \w+ \d{4})", text)
    if date_match:
        try:
            date_parsed = datetime.strptime(date_match.group(1), "%d %B %Y").date().isoformat()
        except ValueError:
            date_parsed = None
    else:
        date_parsed = None

    # Optional: Extract description from Act title
    title_match = re.search(r"(BUILDING CONTROL ACT \d{4}(?:.*?)?)\n", text, re.IGNORECASE)
    description = title_match.group(1).strip() if title_match else os.path.basename(file_path)

    return {
        "location": location,
        "date_iso": date_parsed,
        "description": description
    }


def clean_metadata(metadata: Dict[str, Optional[str]]) -> Dict[str, str]:
    return {k: (str(v) if v is not None else "unknown") for k, v in metadata.items()}

import os

import os

def index_data_from_folder(folder_path: str, target_location: str):
    print(f"\n--- Indexing data from: {folder_path} ---")
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return

    files_processed = 0
    chunks_created = 0

    # # Use os.walk() to traverse subdirectories
    # for root, dirs, files in os.walk(folder_path):
    #     # Check if the current root matches the target_location
    #     if os.path.basename(root) == target_location:
    #         print(f"Found target folder: {root}")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            print('Processing file: ', file_path)  # Debugging print statement

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception as e:
                print(f"Error reading file {filename}: {e}. Skipping.")
                continue

            if not text.strip():
                print(f"Warning: File {filename} is empty. Skipping.")
                continue

            metadata = parse_filename(filename)
            if metadata is None:
                metadata = extract_metadata_from_path(file_path, text)

            chunks = text_splitter.split_text(text)
            embeddings = embedding_model.encode(chunks).tolist()

            ids = []
            metadatas = []
            for i, chunk in enumerate(chunks):
                uid = str(uuid.uuid4())
                ids.append(uid)
                full_metadata = {
                    "text": chunk,
                    "original_filename": filename,
                    "chunk_index": i,
                    **metadata
                }
                metadatas.append(clean_metadata(full_metadata))

            collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )

            files_processed += 1
            chunks_created += len(chunks)

    print(f"Files processed: {files_processed}")
    print(f"Chunks indexed: {chunks_created}")



def retrieve_relevant_context(query: str, top_k: int = 5, location_filter: Optional[str] = None) -> List[Dict]:
    print(f"\n--- Retrieving context for query: '{query}' ---")
    embedding = embedding_model.encode([query])[0].tolist()

    where = {}
    if location_filter:
        where = {
            "$or": [
                {"location": location_filter},
                {"location": "National"},
                {"location": "Unknown"}
            ]
        }

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where=where if location_filter else None
    )

    context = []
    if not results['ids'] or not results['metadatas']:
        print("No relevant results found.")
        return context

    for i in range(len(results['ids'][0])):
        metadata = results['metadatas'][0][i]
        print(f"{i+1}. Location: {metadata.get('location')} | Date: {metadata.get('date_iso')} | File: {metadata.get('original_filename')}")
        context.append(metadata)

    return context


def get_regulatory_assessment(query: str, context: List[Dict]) -> Tuple[Optional[int], Optional[str]]:
    """Sends context to OpenAI to get a regulatory pressure score and explanation."""
    print("\n--- Generating assessment with OpenAI ---")
    if not context:
        print("No context provided to OpenAI.")
        return None, "No relevant context found in the database to assess the query."

    context_string = ""
    sources = set() # Track unique sources
    for i, item in enumerate(context):
        source_info = f"Source {i+1}: [Location: {item['location']}, File: {item['original_filename']}]"
        context_string += f"{source_info}\nContent: {item['text']}\n\n"
        sources.add(f"{item.get('location', 'Unknown Location')} - {item.get('filename', 'Unknown Filename')}")

    prompt = f"""
    You are an AI assistant specialized in analyzing real estate regulations. Your task is to assess the level of regulatory pressure strictly based on the provided legislative context.

    **User Query:** {query}

    **Context from Legislative Documents:**
    --- START CONTEXT ---
    {context_string}
    --- END CONTEXT ---

    **Instructions:**
    1. Review the legislative context carefully in relation to the user query: "{query}".
    2. Assess the level of regulatory pressure based strictly on this context. Consider quantifiable indicators such as:
        - Number and severity of restrictions
        - Specific obligations imposed on landlords or developers
        - Tenant protections (e.g., eviction protections, rent caps)
        - Compliance complexity (e.g., mandatory processes, permits, penalties)
    3. Provide a **Regulatory Pressure Score** from 1 to 10:
        - 1 = Very Low Pressure (few or no restrictions, favorable to landlords/developers)
        - 5 = Moderate Pressure (balanced or standard regulatory obligations)
        - 10 = Very High Pressure (heavy restrictions, strong tenant protections, complex compliance)
    4. Provide a **brief explanation** (2-5 sentences), focused on **quantitative reasoning** (e.g., “The law imposes 6 separate conditions on redevelopment…”).
    5. Identify every legislative article referenced in your explanation, including those marked with **art.**, **act.**, **§**, **article**, or any similar notation. 
    6. Provide **verbatim translations into English** of every legislative article used to justify your score. Only include articles that directly informed your assessment.
    7. **IMPORTANT:** Do NOT use external knowledge. If the provided context does not meaningfully relate to the user query, state that clearly and assign a score of **N/A**.

    **Output Format:**

    Score: [1-10 or N/A]
    Explanation: [Quantitative explanation, grounded in the text]  
    Articles Used (Translated to English):  
    [Full translated text of each cited article]
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
        else:
             print("Warning: Could not parse explanation separately. Returning full AI response as explanation.")

        return score, explanation

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None, f"An error occurred during AI assessment: {e}"

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Ensure Qdrant collection is ready (in memory for this run)

    # 2. Index data into memory (MUST run each time for in-memory)
    # Since the DB is in memory, we need to load data every time the script runs.
    print("\nLoading data into the in-memory Qdrant instance...")
    # Example queries:
    # query = "What are the rules for maximum rent increases?"
    query = "Tenant eviction procedures"
    # query = "Restrictions on maximum rent increase"
    # target_location = None # Search across all locations


    index_data_from_folder(DATA_FOLDER, target_location)


    # 3. Example Query and Assessment
    print("\n" + "="*50)
    print("Example Usage: Assessing Regulatory Pressure (using in-memory DB)")
    print("="*50)


    # a. Retrieve relevant context from memory
    relevant_context = retrieve_relevant_context(query, top_k=5, location_filter=target_location.get('country'))

    # b. Get assessment from OpenAI
    if relevant_context:
        score, explanation = get_regulatory_assessment(query, relevant_context)

        print("\n--- Final Assessment ---")
        print(f"Query: {query}")
        if target_location:
            print(f"Focus Location: {target_location.get('country')}")
        print(f"Retrieved Context Sources (from memory): {len(relevant_context)}")

        if score is not None:
            print(f"Regulatory Pressure Score (1-10): {score}")
        else:
            print("Regulatory Pressure Score (1-10): N/A")

        print(f"\nExplanation:\n{explanation}")
    else:
        print("\n--- Final Assessment ---")
        print(f"Query: {query}")
        print("Could not retrieve relevant context from the in-memory database to perform an assessment.")

    print("\n" + "="*50)
    print("Script finished. In-memory Qdrant data has been discarded.")





