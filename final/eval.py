import os
import re
import uuid
import time
import numpy as np
import pandas as pd
from typing import List, Dict
import random

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import chromadb
from chromadb.config import Settings
import faiss

# --- Load environment variables ---
load_dotenv()

# --- Constants ---
TEXT_CHUNK_SIZE = 500
TEXT_CHUNK_OVERLAP = 100

COUNTRIES = [
    "Australia",
    "China",
    "Japan",
    "New Zealand",
    "Singapore",
    "South Korea"
]

QUERIES = [
    "Legal obligations and procedural requirements for forming, renewing, or terminating a residential lease agreement.",
    "Rules and formalities governing the creation, extension, and cancellation of residential lease contracts.",
    "Procedural and legal steps required to initiate or end a rental agreement, includ\                                ing notice periods and documentation.",
]

EMBEDDING_MODELS = [
    'paraphrase-xlm-r-multilingual-v1'
]

VECTOR_DBS = ["faiss", "chroma", "qdrant"]

# --- Utilities ---
def compute_average_similarity(set1, set2, model):
    emb1 = model.encode(set1, convert_to_tensor=True)
    emb2 = model.encode(set2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).mean().item()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=TEXT_CHUNK_SIZE,
    chunk_overlap=TEXT_CHUNK_OVERLAP,
    length_function=len,
)

def hallucination_score(query: str, retrieved_chunks: List[str]) -> float:
    """Return hallucination score: lower overlap = higher hallucination (0 to 1)."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    query_keywords = set(re.findall(r'\b\w+\b', query.lower())) - ENGLISH_STOP_WORDS
    if not query_keywords:
        return 1.0

    avg_overlap = 0
    for chunk in retrieved_chunks:
        chunk_keywords = set(re.findall(r'\b\w+\b', chunk.lower())) - ENGLISH_STOP_WORDS
        overlap = len(query_keywords & chunk_keywords) / len(query_keywords)
        avg_overlap += overlap
    avg_overlap /= len(retrieved_chunks)

    # Hallucination = 1 - relevance
    return round(1 - avg_overlap, 4)


# --- Vector DB Interfaces ---
class VectorDBBase:
    def index_data(self, chunks, embeddings, metadata): ...
    def retrieve_context(self, query: str, top_k: int = 5): ...
    def reset_index(self): ...

class QdrantDB(VectorDBBase):
    def __init__(self, model):
        self.client = QdrantClient(":memory:")
        self.model = model
        self.collection_name = "real_estate_temp"
        self.vector_size = model.get_sentence_embedding_dimension()

    def reset_index(self):
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except:
            pass
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )

    def index_data(self, chunks, embeddings, metadata):
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i],
                payload=metadata[i]
            )
            for i in range(len(chunks))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def retrieve_context(self, query, top_k=5):
        vec = self.model.encode(query).tolist()
        results = self.client.search(collection_name=self.collection_name, query_vector=vec, limit=top_k, with_payload=True)
        return [{**r.payload, 'score': r.score} for r in results]

class ChromaDB(VectorDBBase):
    def __init__(self, model):
        self.client = chromadb.Client(Settings())
        self.model = model
        self.collection_name = "real_estate_temp"
        self.collection = None

    def reset_index(self):
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass
        self.collection = self.client.create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def index_data(self, chunks, embeddings, metadata):
        ids = [str(uuid.uuid4()) for _ in chunks]
        self.collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadata)

    def retrieve_context(self, query, top_k=5):
        vec = self.model.encode([query])[0].tolist()
        results = self.collection.query(query_embeddings=[vec], n_results=top_k, include=["metadatas", "distances"])
        return [{**meta, 'score': dist} for meta, dist in zip(results['metadatas'][0], results['distances'][0])]

class FAISSDB(VectorDBBase):
    def __init__(self, model):
        self.model = model
        self.index = None
        self.metadata = {}
        self.next_id = 0

    def reset_index(self):
        dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = {}
        self.next_id = 0

    def index_data(self, chunks, embeddings, metadata):
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)  # Normalize for cosine
        self.index.add(vectors)
        start_id = self.index.ntotal - len(vectors)
        for i, meta in enumerate(metadata):
            self.metadata[start_id + i] = meta

    def retrieve_context(self, query, top_k=5):
        vec = self.model.encode([query]).astype(np.float32)
        faiss.normalize_L2(vec)
        D, I = self.index.search(vec, top_k)
        return [{**self.metadata[i], 'score': float(D[0][j])} for j, i in enumerate(I[0]) if i in self.metadata]

# --- Main Loop ---
all_country_results = []

for location in COUNTRIES:
    data_folder = os.path.join("txt", location)
    print(f"Processing location: {location}")

    location_results = []

    for model_name in EMBEDDING_MODELS:
        model = SentenceTransformer(model_name)
        print(f"Loaded model: {model_name}")

        # Embed once per location/model
        chunks = []
        metadata = []
        embedding_time = 0
        start_time = time.time()
        for filename in os.listdir(data_folder):
            print(f"Processing file: {filename}...")
            if filename.endswith(".txt"):
                with open(os.path.join(data_folder, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                file_chunks = text_splitter.split_text(text)
                for i, chunk in enumerate(file_chunks):
                    chunks.append(chunk)
                    metadata.append({"text": chunk, "original_filename": filename, "country": location, "chunk_index": i})

        embeddings = model.encode(chunks).tolist()
        embedding_time += (time.time() - start_time)

        for db_type in VECTOR_DBS:
            print(f"Evaluating {model_name} with {db_type} on {location}...")
            if db_type == "qdrant":
                db = QdrantDB(model)
            elif db_type == "chroma":
                db = ChromaDB(model)
            elif db_type == "faiss":
                db = FAISSDB(model)
            else:
                continue

            db.reset_index()
            db.index_data(chunks, embeddings, metadata)

            context_sets = {}
            total_time = 0
            count = 0
            hallucination_scores = []

            for q in QUERIES:
                start = time.time()
                context = db.retrieve_context(q)
                total_time += (time.time() - start)
                print(f"Top results for '{q}' from {db_type}:")
                for idx, c in enumerate(context):
                    print(f"  {idx+1}. {c.get('original_filename', 'N/A')} | Chunk: {c.get('chunk_index', 'N/A')} | Score: {round(c.get('score', 0), 4)}")
                texts = [c['text'] for c in context]
                context_sets[q] = texts
                halluc_score = hallucination_score(q, texts)
                hallucination_scores.append(halluc_score)

                print(f"  Hallucination score: {halluc_score}")

                count += 1

            # Compare all queries to the first one
            reference = context_sets[QUERIES[0]]
            pair_scores = []
            for i in range(1, len(QUERIES)):
                score = compute_average_similarity(reference, context_sets[QUERIES[i]], model)
                pair_scores.append(score)
            inter_score = sum(pair_scores) / len(pair_scores)


            avg_hallucination = sum(hallucination_scores) / len(hallucination_scores)

            location_results.append({
                "Location": location,
                "Model": model_name,
                "Vector DB": db_type,
                "Similarity": round(inter_score, 4),
                "Hallucination": round(avg_hallucination, 4),
                "Time (s)": round(total_time / count, 3)
            })


    all_country_results.extend(location_results)

# --- Aggregate and Save Results ---



df = pd.DataFrame(all_country_results)
avg_df = df.groupby(["Model", "Vector DB"]).agg({
    "Similarity": "mean",
    "Hallucination": "mean",
    "Time (s)": "mean"
}).reset_index()
avg_df.to_csv("model_db_similarity_results.csv", index=False)
print("Evaluation complete. Results saved to model_db_similarity_results.csv")