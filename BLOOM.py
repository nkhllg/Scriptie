import os
import re
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# NEW: Imports for BLOOM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Configuration ---
load_dotenv()

DATA_FOLDER = "legislation_data"
COLLECTION_NAME = "real_estate_regulations_temp"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
TEXT_CHUNK_SIZE = 1000
TEXT_CHUNK_OVERLAP = 150

# Filename parsing regex
FILENAME_PATTERN = re.compile(r"^(?P<location>.+?)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<desc>.*)\.txt$", re.IGNORECASE)

# --- Initialization ---
print("Initializing components...")

# Load embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
vector_size = embedding_model.get_sentence_embedding_dimension()

# Initialize Qdrant Client (in-memory)
qdrant_client = QdrantClient(":memory:")

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=TEXT_CHUNK_SIZE,
    chunk_overlap=TEXT_CHUNK_OVERLAP,
    length_function=len,
)

# NEW: Load BLOOM model (may require GPU or large memory)
print("Loading BLOOM model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
bloom_model_name = "bigscience/bloom-1b7"
tokenizer = AutoTokenizer.from_pretrained(bloom_model_name)
model = AutoModelForCausalLM.from_pretrained(bloom_model_name).to(device)
print("BLOOM model loaded.")

# --- Qdrant Setup ---
def setup_qdrant_collection():
    try:
        qdrant_client.get_collection(collection_name=COLLECTION_NAME)
    except Exception:
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        qdrant_client.create_payload_index(COLLECTION_NAME, "location", models.PayloadSchemaType.KEYWORD)
        qdrant_client.create_payload_index(COLLECTION_NAME, "date_iso", models.PayloadSchemaType.KEYWORD)

# --- Parsing and Indexing ---
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

def index_data_from_folder(folder_path: str):
    points_to_upload = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            metadata = parse_filename(filename) or {
                "location": "Unknown", "date_iso": None, "description": filename
            }
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            if not text.strip():
                continue

            chunks = text_splitter.split_text(text)
            embeddings = embedding_model.encode(chunks).tolist()

            for i, chunk in enumerate(chunks):
                payload = {
                    "text": chunk,
                    "original_filename": filename,
                    "location": metadata.get("location", "Unknown"),
                    "date_iso": metadata.get("date_iso"),
                    "description": metadata.get("description", "Unknown"),
                    "chunk_index": i
                }
                payload = {k: v for k, v in payload.items() if v is not None}
                points_to_upload.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[i],
                    payload=payload
                ))

            if len(points_to_upload) >= 100:
                qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points_to_upload, wait=True)
                points_to_upload = []

    if points_to_upload:
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points_to_upload, wait=True)

# --- Retrieval ---
def retrieve_relevant_context(query: str, top_k: int = 5, location_filter: Optional[str] = None) -> List[Dict]:
    query_vector = embedding_model.encode(query).tolist()

    filter_conditions = []
    if location_filter:
        filter_conditions.extend([
            models.FieldCondition(key="location", match=models.MatchValue(value=location_filter)),
            models.FieldCondition(key="location", match=models.MatchValue(value="National")),
            models.FieldCondition(key="location", match=models.MatchValue(value="Unknown")),
        ])
    qdrant_filter = models.Filter(should=filter_conditions) if filter_conditions else None

    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True
    )

    return [{
        "text": hit.payload.get("text"),
        "location": hit.payload.get("location"),
        "date": hit.payload.get("date_iso"),
        "filename": hit.payload.get("original_filename"),
        "score": hit.score
    } for hit in search_result]

# --- BLOOM-Based Assessment ---
def get_regulatory_assessment(query: str, context: List[Dict]) -> Tuple[Optional[int], Optional[str]]:
    if not context:
        return None, "No relevant context found."

    context_string = ""
    for i, item in enumerate(context):
        source_info = f"Source {i+1}: [Location: {item['location']}, Date: {item['date']}, File: {item['filename']}]"
        context_string += f"{source_info}\nContent: {item['text']}\n\n"

    prompt = f"""
You are an expert on real estate regulations.

User Query: {query}

--- CONTEXT START ---
{context_string}
--- CONTEXT END ---

Based strictly on the context above:
- Assign a Regulatory Pressure Score (1 = very low, 10 = very high, N/A if not enough info).
- Give a short explanation citing sources (e.g., "Source 2").

Respond in this format:
Score: [1-10 or N/A]
Explanation: [your reasoning]
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    output = model.generate(**inputs, max_new_tokens=300, temperature=0.3, do_sample=True)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract only the final answer (after "Score:")
    score_match = re.search(r"Score:\s*(\d+|N/A)", decoded, re.IGNORECASE)
    explanation_match = re.search(r"Explanation:\s*(.*)", decoded, re.IGNORECASE | re.DOTALL)

    score = None
    explanation = decoded.strip()
    if score_match:
        score_str = score_match.group(1)
        score = int(score_str) if score_str.isdigit() else None
    if explanation_match:
        explanation = explanation_match.group(1).strip()

    return score, explanation

# --- Main ---
if __name__ == "__main__":
    setup_qdrant_collection()
    index_data_from_folder(DATA_FOLDER)

    query = "Restrictions on maximum rent increase"
    location = "Korea"

    context = retrieve_relevant_context(query, top_k=5, location_filter=location)
    score, explanation = get_regulatory_assessment(query, context)

    print("\n--- Final Assessment ---")
    print(f"Query: {query}")
    print(f"Location: {location}")
    print(f"Score: {score if score is not None else 'N/A'}")
    print(f"Explanation: {explanation}")
