import os
import re
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import hashlib

import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
from geolocation import get_location_nominatim

# address = "53 Forbes st, devonport"
# target_location = get_location_nominatim(address)
target_location = {'city': '', 'province': '', 'country': 'Japan'}

print('target location: ', target_location)

COUNTRY = re.sub(r'[^a-zA-Z0-9_-]', '_', target_location.get('country', '').strip().lower())

# --- Configuration ---
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")


# EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
EMBEDDING_MODEL = 'intfloat/e5-base'
# EMBEDDING_MODEL = 'nlpaueb/legal-bert-base-uncased'
# EMBEDDING_MODEL = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"  #512
# EMBEDDING_MODEL = 'distiluse-base-multilingual-cased-v2'
# EMBEDDING_MODEL = 'sentence-transformers/LaBSE'
TEXT_CHUNK_SIZE = 500
TEXT_CHUNK_OVERLAP = 150


embedding_model = SentenceTransformer(EMBEDDING_MODEL)
vector_size = embedding_model.get_sentence_embedding_dimension()
COLLECTION_NAME = f"real_estate_regulations_{COUNTRY}_dim{vector_size}"

# Use persistent client instead of in-memory
chroma_client = chromadb.PersistentClient(path="chroma_db_data")
# collection = chroma_client.get_or_create_collection(
#     COLLECTION_NAME,
#     metadata={"hnsw:space": "cosine"}
# )

# === COUNTRY-SPECIFIC SETUP ===
if COUNTRY.lower() == "australia":
    province = target_location.get('province', '')
    # Call your custom setup function for Australia
    DATA_FOLDER = os.path.join("txt", "Australia", province)
    province = province.replace(' ', '_')
    FILENAME_PATTERN = re.compile(
        r"^(?P<location>.+?)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<desc>.*)\.txt$",
        re.IGNORECASE
    )
    COLLECTION_NAME = f"real_estate_regulations_Australia_{province}_dim{vector_size}"

else:
    # Fallback for other countries — default folder structure
    DATA_FOLDER = os.path.join("txt", COUNTRY)
    FILENAME_PATTERN = re.compile(
        r"^(?P<location>.+?)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<desc>.*)\.txt$",
        re.IGNORECASE
    )
    COLLECTION_NAME = f"real_estate_regulations_{COUNTRY}_dim{vector_size}"

collection = chroma_client.get_or_create_collection(
    COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=TEXT_CHUNK_SIZE,
    chunk_overlap=TEXT_CHUNK_OVERLAP,
    length_function=len,
)

def get_file_hash(filepath):
    """Generate MD5 hash of file contents"""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

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

def is_location_indexed(collection, country, province=None, city=None):
    # Example filter dict - adjust depending on your collection's metadata schema
    filter_dict = {"country": country}
    if province:
        filter_dict["province"] = province
    if city:
        filter_dict["city"] = city

    # Query the collection for items matching this filter
    matching_items = collection.query(filter=filter_dict, n_results=1)
    return len(matching_items["ids"]) > 0


def extract_metadata_from_path(file_path: str, text: str) -> Dict[str, str]:
    """Extract metadata using the file path (for location) and content (for optional date/description)."""
    location = os.path.basename(os.path.dirname(file_path)) or "Unknown"

    date_match = re.search(r"(?:operation on|amendments up to and including)\s+(\d{1,2} \w+ \d{4})", text)
    if date_match:
        try:
            date_parsed = datetime.strptime(date_match.group(1), "%d %B %Y").date().isoformat()
        except ValueError:
            date_parsed = None
    else:
        date_parsed = None

    title_match = re.search(r"(BUILDING CONTROL ACT \d{4}(?:.*?)?)\n", text, re.IGNORECASE)
    description = title_match.group(1).strip() if title_match else os.path.basename(file_path)

    return {
        "location": location,
        "date_iso": date_parsed,
        "description": description
    }

def clean_metadata(metadata: Dict[str, Optional[str]]) -> Dict[str, str]:
    return {k: (str(v) if v is not None else "unknown") for k, v in metadata.items()}

def index_data_from_folder(folder_path: str, target_location: str, force_reindex: bool = False):
    print(f"\n--- Indexing data from: {folder_path} ---")
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return

    files_processed = 0
    chunks_created = 0

    # Get existing documents metadata
    existing_docs = collection.get()
    existing_files = {}
    if existing_docs and 'metadatas' in existing_docs:
        for i, meta in enumerate(existing_docs['metadatas']):
            if 'original_filename' in meta:
                existing_files[meta['original_filename']] = {
                    'hash': meta.get('file_hash', ''),
                    'ids': [existing_docs['ids'][i]] if i < len(existing_docs['ids']) else []
                }

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".txt"):
            continue

        file_path = os.path.join(folder_path, filename)
        current_hash = get_file_hash(file_path)

        # Skip if file exists and hashes match (unless force_reindex)
        if filename in existing_files:
            if not force_reindex and existing_files[filename]['hash'] == current_hash:
                print(f"Skipping unchanged file: {filename}")
                continue
            # Delete old version if reindexing
            if existing_files[filename]['ids']:
                collection.delete(ids=existing_files[filename]['ids'])

        print(f'Processing file: {file_path}')

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
                "file_hash": current_hash,
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
    sources = set()
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
    4. Provide a **brief explanation** (2-5 sentences), focused on **quantitative reasoning** (e.g., "The law imposes 6 separate conditions on redevelopment...").
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
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in analyzing real estate regulations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        ai_message = response.choices[0].message.content.strip()
        print("OpenAI Assessment Received.")

        score = None
        explanation = ai_message

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
                    score = None

        if explanation_match:
            explanation = explanation_match.group(1).strip()
        else:
             print("Warning: Could not parse explanation separately. Returning full AI response as explanation.")

        return score, explanation

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None, f"An error occurred during AI assessment: {e}"

def evaluate_ai_assessment(query: str, context: List[Dict], ai_response: str) -> str:
    """Evaluates the quality of the AI-generated assessment using a meta-prompt."""
    print("\n--- Evaluating AI Response Quality ---")

    context_string = ""
    for i, item in enumerate(context):
        source_info = f"Source {i+1}: [Location: {item['location']}, File: {item['original_filename']}]"
        context_string += f"{source_info}\nContent: {item['text']}\n\n"

    evaluation_prompt = f"""
You are a legal expert and AI evaluator. Your job is to assess how well the following AI response answers the user's query using only the legislative context provided.

---

**User Query:**  
{query}

**Context Provided to the AI:**  
--- START CONTEXT ---  
{context_string}  
--- END CONTEXT ---

**AI's Previous Response:**  
{ai_response}

---

**Evaluation Criteria:**
1. Is the explanation grounded in specific, quoted or paraphrased legislative context?
2. Are actual legislative articles cited and translated where required?
3. Is the reasoning quantitative and relevant?
4. Does the response follow all instructions without hallucination?

---

**Your Evaluation Output:**

Evaluation Score (1–10): [1 = very poor, 10 = excellent]  
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a strict evaluator of AI legal assessments."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        evaluation = response.choices[0].message.content.strip()
        print("\n--- Evaluation Complete ---")
        return evaluation

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return "Evaluation failed due to an error."

def clean_index():
    """Remove all documents from the collection"""
    ids = collection.get()['ids']
    if ids:
        collection.delete(ids=ids)
    print("Index cleared")


if __name__ == "__main__":
    # Example query
    # query = "Restrictions on maximum rent increase"
    query = "Tenant eviction procedures"
    
    # Check if we need to index (collection is empty)
    existing_count = collection.count()
    if existing_count == 0:
        print("\nInitial data load into Chroma...")
        index_data_from_folder(DATA_FOLDER, target_location)
    else:
        print(f"\nUsing existing Chroma collection with {existing_count} items")
        # Optional: Uncomment to force reindex if needed
        # print("Forcing reindex...")
        # clean_index()
        # index_data_from_folder(DATA_FOLDER, target_location, force_reindex=True)

    print("\n" + "="*50)
    print("Example Usage: Assessing Regulatory Pressure")
    print("="*50)

    relevant_context = retrieve_relevant_context(query, top_k=5, location_filter=target_location.get('country'))

    if relevant_context:
        score, explanation = get_regulatory_assessment(query, relevant_context)

        print("\n--- Final Assessment ---")
        print(f"Query: {query}")
        if target_location:
            print(f"Focus Location: {target_location}")
        print(f"Retrieved Context Sources: {len(relevant_context)}")

        if score is not None:
            print(f"Regulatory Pressure Score (1-10): {score}")
        else:
            print("Regulatory Pressure Score (1-10): N/A")

        print(f"\nExplanation:\n{explanation}")

        evaluation = evaluate_ai_assessment(query, relevant_context, explanation)
        print("\n--- Evaluation of AI's Answer ---")
        print(evaluation)
    else:
        print("\n--- Final Assessment ---")
        print(f"Query: {query}")
        print("Could not retrieve relevant context from the database to perform an assessment.")

    print("\n" + "="*50)
    print("Script finished. Chroma data persists for future runs.")
