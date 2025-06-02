import os
import re
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import requests
import time
import json
import pandas as pd

import openai
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from geolocation import get_location_nominatim

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file

# --- Core Functions ---

def setup_qdrant_collection():
    """Deletes and recreates the Qdrant collection for a fresh start."""
    try:
        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'.")
    except Exception as e:
        print(f"No existing collection to delete or error occurred: {e}")

    print(f"Creating fresh in-memory collection '{COLLECTION_NAME}'...")
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print("In-memory collection created.")

def index_data_from_folder(folder_path: str):
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
                    "country": country,
                    "chunk_index": i
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


def retrieve_relevant_context(query: str, top_k: int = 5) -> List[Dict]:
    """Embeds query and retrieves relevant documents from the in-memory Qdrant."""
    print(f"\n--- Retrieving context from memory for query: '{query}' ---")

    query_vector = embedding_model.encode(query).tolist()

    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True  # Ensure payload (text, metadata) is returned
    )

    context = []
    print(f"Found {len(search_result)} relevant chunks in memory:")
    for i, hit in enumerate(search_result):
        payload = hit.payload
        context_text = payload.get('text', 'N/A')
        location = payload.get('country', 'N/A')
        filename = payload.get('original_filename', 'N/A')
        print(f"  {i+1}. Score: {hit.score:.4f} | Location: {location} | File: {filename}")
        # print(f"     Text: {context_text[:150]}...") # Print snippet
        context.append({
            "text": context_text,
            "location": location,
            "filename": filename,
            "score": hit.score
        })
    return context

def compute_average_similarity(set1, set2):
    # Flatten and embed
    emb1 = embedding_model.encode(set1, convert_to_tensor=True)
    emb2 = embedding_model.encode(set2, convert_to_tensor=True)

    # Compute pairwise cosine similarities
    cosine_scores = util.cos_sim(emb1, emb2)
    
    # Return average similarity
    return cosine_scores.mean().item()


# --- Main Execution ---
if __name__ == "__main__":
    COLLECTION_NAME = "real_estate_regulations_temp" # Use a distinct name maybe
    TEXT_CHUNK_SIZE = 500  # Characters per chunk
    TEXT_CHUNK_OVERLAP = 100 # Overlap between chunks

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_CHUNK_SIZE,
        chunk_overlap=TEXT_CHUNK_OVERLAP,
        length_function=len,
    )

    qdrant_client = QdrantClient(":memory:")

    model_names = [
        "paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
        "sentence-transformers/LaBSE"
    ]

    locations = ["Austria", "Denmark", "Finland", "France", "Ireland", "Italy", "Netherlands", "Norway", "Portugal", "Spain", "Sweden"]

    queries = [
        [
            "Restrictions on maximum rent increase",
            "Government-imposed caps on rent increases",
            "Government regulations on raising rent",
        ],
        [
            "Tenant eviction procedures",
            "Government rules for evicting tenants",
            "Regulations on tenant eviction process",
        ],
        [
            "Requirements for obtaining building permits",
            "Rules governing building permit approvals",
            "Regulations for securing construction permits",
        ],
    ]

    hallucination_queries = [
        "Popular travel destinations in spring",
        "Best practices for learning a new language",
        "Benefits of daily morning walks"
    ]

    results = []

    for EMBEDDING_MODEL in model_names:
        # Initialize Sentence Transformer model
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        vector_size = embedding_model.get_sentence_embedding_dimension()
        print(f"Embedding model loaded. Vector size: {vector_size}")

        scores = []
        inter_scores = []
        hallucination_scores = []
        total_time = 0
        count = 0

        for country in locations:
            DATA_FOLDER = os.path.join("..", "Data_txts", country.strip())
            FILENAME_PATTERN = re.compile(r"^(?P<location>.+?)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<desc>.*)\.txt$", re.IGNORECASE)

            setup_qdrant_collection()
            start_time = time.time()
            index_data_from_folder(DATA_FOLDER)
            end_time = time.time()
            embedding_time = end_time - start_time

            context_sets_per_query = {}

            for query_group in queries:
                for subquery in query_group:
                    relevant_context = retrieve_relevant_context(subquery, top_k=5)
                    context_texts = [c['text'] for c in relevant_context]
                    context_sets_per_query[subquery] = context_texts

                    score = [c['score'] for c in relevant_context]
                    combined_score = sum(score) / len(score)
                    scores.append(combined_score)

                reference = context_sets_per_query[query_group[0]]
                for subquery in query_group[1:]:
                    score = compute_average_similarity(reference, context_sets_per_query[subquery])
                    inter_scores.append(score)

            for h_query in hallucination_queries:

                hallucinated_context = retrieve_relevant_context(h_query, top_k=5)

                hallucinated_scores = [c['score'] for c in hallucinated_context]
                hallucination_score = sum(hallucinated_scores) / len(hallucinated_scores)
                hallucination_scores.append(hallucination_score)

        avg_score = sum(scores) / len(scores)
        avg_inter_score = sum(inter_scores) / len(inter_scores)
        avg_hallucination = sum(hallucination_scores) / len(hallucination_scores)
        avg_time_per_query = end_time - start_time

        results.append({
            "Model": EMBEDDING_MODEL,
            "Vector DB": "qdrant",
            "score": round(avg_score, 4),
            "Similarity": round(avg_inter_score, 4),
            "Hallucination": round(avg_hallucination, 4),
            "Time (s)": round(avg_time_per_query, 4)
        })

    df = pd.DataFrame(results)
    df.to_csv("embedding_scores.csv", index=False)