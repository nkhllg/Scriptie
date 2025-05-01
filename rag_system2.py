import os
import re
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import requests
import time

import openai
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from geolocation import get_location_nominatim

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file

DATA_FOLDER = "legislation_data"  # Folder containing your text files
COLLECTION_NAME = "real_estate_regulations_temp" # Use a distinct name maybe
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Efficient and good quality
TEXT_CHUNK_SIZE = 1000  # Characters per chunk
TEXT_CHUNK_OVERLAP = 150 # Overlap between chunks

# OpenAI Configuration
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Filename parsing regex (adjust if your naming convention is different)
# Example: Busan_2023-05-10_publicLandUseAct.txt
FILENAME_PATTERN = re.compile(r"^(?P<location>.+?)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<desc>.*)\.txt$", re.IGNORECASE)

# --- Initialization ---
print("Initializing components...")
# Initialize Sentence Transformer model
print(f"Loading embedding model: {EMBEDDING_MODEL}...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
vector_size = embedding_model.get_sentence_embedding_dimension()
print(f"Embedding model loaded. Vector size: {vector_size}")

# Initialize Qdrant Client for IN-MEMORY storage
print("Initializing Qdrant client for in-memory storage...")
qdrant_client = QdrantClient(":memory:")
print("Qdrant running in-memory. Data will not be persisted after script exits.")


# Initialize Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=TEXT_CHUNK_SIZE,
    chunk_overlap=TEXT_CHUNK_OVERLAP,
    length_function=len,
)
print("Components initialized.")

# --- Core Functions ---

def setup_qdrant_collection():
    """Creates the Qdrant collection if it doesn't exist (in memory)."""
    try:
        # Check if collection exists in the current in-memory instance
        qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists in this session.")
        # Optionally recreate it if you want a fresh start even within the same script run
        # print(f"Recreating collection '{COLLECTION_NAME}' in memory...")
        # qdrant_client.recreate_collection(...)
    except Exception:
        print(f"Creating in-memory collection '{COLLECTION_NAME}'...")
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            # In-memory specific optimizations might be less critical, but schema is good
            # optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
            # hnsw_config=models.HnswConfigDiff(on_disk=False, m=16, ef_construct=100) # on_disk=False is implicit for :memory:
        )
        print("In-memory collection created.")
        # Optional: Create payload indexes for filtering (still useful in memory)
        qdrant_client.create_payload_index(COLLECTION_NAME, field_name="location", field_schema=models.PayloadSchemaType.KEYWORD)
        qdrant_client.create_payload_index(COLLECTION_NAME, field_name="Region", field_schema=models.PayloadSchemaType.KEYWORD)
        qdrant_client.create_payload_index(COLLECTION_NAME, field_name="City", field_schema=models.PayloadSchemaType.KEYWORD)
        qdrant_client.create_payload_index(COLLECTION_NAME, field_name="Building_type", field_schema=models.PayloadSchemaType.KEYWORD)
        qdrant_client.create_payload_index(COLLECTION_NAME, field_name="date_iso", field_schema=models.PayloadSchemaType.KEYWORD) # Store date as ISO string
        print("Payload indexes created.")


def parse_filename(filename: str) -> Optional[Dict[str, str]]:
    """Parses location and date from filename using regex."""
    match = FILENAME_PATTERN.match(filename)
    if match:
        data = match.groupdict()
        try:
            # Validate date format
            datetime.strptime(data['date'], '%Y-%m-%d')
            return {
                "location": data['location'].replace('_', ' ').strip(), # Replace underscores if used
                "date_iso": data['date'],
                "description": data['desc'].replace('_', ' ').strip()
            }
        except ValueError:
            print(f"Warning: Invalid date format in filename '{filename}'. Skipping metadata date.")
            return { # Still return location if possible
                 "location": data['location'].replace('_', ' ').strip(),
                 "date_iso": None,
                 "description": data['desc'].replace('_', ' ').strip()
             }
    else:
        print(f"Warning: Filename '{filename}' did not match expected pattern. Skipping metadata extraction.")
        return None

def index_data_from_folder(folder_path: str, city, region, target_location):
    """Reads files, chunks text, embeds, and uploads to the in-memory Qdrant."""
    print(f"\n--- Starting data indexing into memory from folder: {folder_path} ---")
    points_to_upload = []
    files_processed = 0
    chunks_created = 0

    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {filename}...")
            metadata = parse_filename(filename)
            if metadata is None:
                # Assign default metadata if parsing fails but still process the file
                 metadata = {"location": "Unknown", "date_iso": None, "description": filename}

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception as e:
                print(f"Error reading file {filename}: {e}. Skipping.")
                continue

            if not text.strip():
                print(f"Warning: File {filename} is empty. Skipping.")
                continue

            # Chunk the text
            chunks = text_splitter.split_text(text)
            print(f"  - Created {len(chunks)} chunks.")
            chunks_created += len(chunks)

            # Embed chunks
            print(f"  - Embedding {len(chunks)} chunks...")
            try:
                embeddings = embedding_model.encode(chunks, show_progress_bar=False).tolist()
            except Exception as e:
                print(f"Error embedding chunks for file {filename}: {e}. Skipping file.")
                continue

            # Prepare points for Qdrant
            for i, chunk in enumerate(chunks):
                point_id = str(uuid.uuid4()) # Generate unique ID for each chunk
                payload = {
                    "text": chunk,
                    "original_filename": filename,
                    "location": target_location,
                    "region": region,
                    "city": city,
                    "building_type": "apartment", # change to a function later and make sure it is incorporated!
                    "date_iso": metadata.get("date_iso"),
                    "description": metadata.get("description", "Unknown"),
                    "chunk_index": i # Optional: track chunk order within file
                }
                # Filter out None values from payload before creating PointStruct
                payload = {k: v for k, v in payload.items() if v is not None}

                points_to_upload.append(PointStruct(
                    id=point_id,
                    vector=embeddings[i],
                    payload=payload
                ))

            files_processed += 1

            # Upload in batches (still good practice even in memory)
            if len(points_to_upload) >= 100:
                print(f"Adding batch of {len(points_to_upload)} points to memory...")
                qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points_to_upload, wait=True)
                points_to_upload = []

    # Upload any remaining points
    if points_to_upload:
        print(f"Adding final batch of {len(points_to_upload)} points to memory...")
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points_to_upload, wait=True)

    print(f"\n--- Indexing to memory complete ---")
    print(f"Files processed: {files_processed}")
    print(f"Total chunks created and indexed in memory: {chunks_created}")
    # Verify count in Qdrant
    count = qdrant_client.count(collection_name=COLLECTION_NAME, exact=True)
    print(f"Total points in in-memory collection '{COLLECTION_NAME}': {count.count}")


def retrieve_relevant_context(query: str, top_k: int = 5, location_filter: Optional[str] = None) -> List[Dict]:
    """Embeds query and retrieves relevant documents from the in-memory Qdrant."""
    print(f"\n--- Retrieving context from memory for query: '{query}' ---")
    if location_filter:
        print(f"Applying location filter: '{location_filter}'")

    query_vector = embedding_model.encode(query).tolist()

    filter_conditions = []
    if location_filter:
        # Find docs matching the specific location OR National OR Unknown
        filter_conditions.append(
            models.FieldCondition(
                key="location",
                match=models.MatchValue(value=location_filter)
            )
        )
        filter_conditions.append(
             models.FieldCondition(key="location", match=models.MatchValue(value="National"))
        )
        filter_conditions.append(
             models.FieldCondition(key="location", match=models.MatchValue(value="Unknown"))
        )

    # Use a "should" filter (OR logic) if location filter is applied
    qdrant_filter = models.Filter(should=filter_conditions) if filter_conditions else None

    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True  # Ensure payload (text, metadata) is returned
    )

    context = []
    print(f"Found {len(search_result)} relevant chunks in memory:")
    for i, hit in enumerate(search_result):
        payload = hit.payload
        context_text = payload.get('text', 'N/A')
        location = payload.get('location', 'N/A')
        date = payload.get('date_iso', 'N/A')
        filename = payload.get('original_filename', 'N/A')
        print(f"  {i+1}. Score: {hit.score:.4f} | Location: {location} | Date: {date} | File: {filename}")
        # print(f"     Text: {context_text[:150]}...") # Print snippet
        context.append({
            "text": context_text,
            "location": location,
            "date": date,
            "filename": filename,
            "score": hit.score
        })
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
        source_info = f"Source {i+1}: [Location: {item['location']}, Date: {item['date']}, File: {item['filename']}]"
        context_string += f"{source_info}\nContent: {item['text']}\n\n"
        sources.add(f"{item['location']} - {item['filename']} ({item['date']})")

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
    # 1. Ensure Qdrant collection is ready (in memory for this run)
    setup_qdrant_collection()

    address = "Burgemeester Fockema Andreaelaan, 3582"
    city, region, target_location = get_location_nominatim(address)

    # 2. Index data into memory (MUST run each time for in-memory)
    # Since the DB is in memory, we need to load data every time the script runs.
    print("\nLoading data into the in-memory Qdrant instance...")
    index_data_from_folder(DATA_FOLDER, city, region, target_location)


    # 3. Example Query and Assessment
    print("\n" + "="*50)
    print("Example Usage: Assessing Regulatory Pressure (using in-memory DB)")
    print("="*50)

    # Example queries:
    # query = "What are the rules for maximum rent increases?"
    # query = "Tenant eviction procedures"
    query = "Restrictions on maximum rent increase"
    # target_location = None # Search across all locations

    # a. Retrieve relevant context from memory
    relevant_context = retrieve_relevant_context(query, top_k=5, location_filter=target_location)

    # b. Get assessment from OpenAI
    if relevant_context:
        score, explanation = get_regulatory_assessment(query, relevant_context)

        print("\n--- Final Assessment ---")
        print(f"Query: {query}")
        if target_location:
            print(f"Focus Location: {target_location}")
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





