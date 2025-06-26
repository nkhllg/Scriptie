import os
import re
import time
import uuid
import glob
import pandas as pd
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.models import SearchParams
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
model_names = [
    'all-MiniLM-L6-v2',
    'nlpaueb/legal-bert-base-uncased',
    'intfloat/e5-base',
    "paraphrase-xlm-r-multilingual-v1",
    'distiluse-base-multilingual-cased-v2',
    'sentence-transformers/LaBSE'
]

locations = ["South Korea", "Singapore", "New Zealand", "China", "Japan", "Australia"]
countries_slug = [re.sub(r'[^a-zA-Z0-9_-]', '_', loc.strip()) for loc in locations]

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

def safe_collection_name(prefix: str, model_name: str, country: str, suffix: str = "", max_length: int = 63) -> str:
    # Replace invalid characters in model name
    model_safe = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
    
    # Compose preliminary name
    base = f"{prefix}_{model_safe}_{country}_{suffix}"
    
    # If too long, truncate the model part
    if len(base) > max_length:
        # Calculate max length allowed for model part
        max_model_length = max_length - len(prefix) - len(suffix) - 2  # 2 underscores
        model_safe = model_safe[:max_model_length]
        base = f"{prefix}_{model_safe}_{suffix}"

    return base

# def get_collection_name(country_slug: str, vector_size: int) -> str:
#     """Generate collection name matching the cache version's naming convention"""
#     return f"real_estate_regulations_{country_slug}_dim{vector_size}"


def split_text_into_chunks(text: str, chunk_size=500, overlap=100) -> List[str]:
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def ensure_collection_and_load_documents(qdrant_client: QdrantClient, collection_name: str, country: str,
                                         embed_model: SentenceTransformer, chunk_size=500, overlap=100):
    existing_collections = [c.name for c in qdrant_client.get_collections().collections]
    if collection_name not in existing_collections:
        print(f"Creating collection: {collection_name}")
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embed_model.get_sentence_embedding_dimension(), distance=Distance.COSINE)
        )

        country_path = os.path.join("txt", country)
        txt_files = sorted(glob.glob(os.path.join(country_path, "*.txt")))

        if not txt_files:
            print(f"No documents found for country: {country}")
            return

        points = []
        for file_path in txt_files:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
            embeddings = embed_model.encode(chunks, show_progress_bar=False)

            for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "country": country,
                        "file": os.path.basename(file_path),
                        "chunk_index": i,
                        "text": chunk_text
                    }
                ))

        qdrant_client.upsert(collection_name=collection_name, points=points)
        print(f"Uploaded {len(points)} chunks to collection '{collection_name}'.")
    else:
        print(f"Collection '{collection_name}' already exists. Skipping creation.")


def compute_average_similarity(texts1: List[str], texts2: List[str], embedding_model: SentenceTransformer) -> float:
    emb1 = embedding_model.encode(texts1, convert_to_tensor=True, show_progress_bar=False)
    emb2 = embedding_model.encode(texts2, convert_to_tensor=True, show_progress_bar=False)
    cosine_scores = util.cos_sim(emb1, emb2)
    return cosine_scores.mean().item()


def evaluate_model(embedding_model_name: str, qdrant_client: QdrantClient) -> Dict:
    print(f"\nLoading embedding model: {embedding_model_name}...")
    model = SentenceTransformer(embedding_model_name)
    vector_size = model.get_sentence_embedding_dimension()
    print(f"Embedding model loaded. Vector size: {vector_size}")

    prefix = "RE_reg"
    scores = []
    inter_scores = []
    hallucination_scores = []
    total_time = 0
    count = 0

    for country_slug, location in zip(countries_slug, locations):
        collection_name = safe_collection_name(prefix, embedding_model_name, country_slug, suffix=f"dim{vector_size}")

        # Ensure collection exists and data is loaded
        try:
            ensure_collection_and_load_documents(
                qdrant_client=qdrant_client,
                collection_name=collection_name,
                country=location,
                embed_model=model,
                chunk_size=500,
                overlap=100
            )
        except Exception as e:
            print(f"Error preparing collection {collection_name}: {e}. Skipping.")
            continue

        context_sets_per_query = {}

        for query_group in queries:
            for subquery in query_group:
                query_vector = model.encode(subquery, show_progress_bar=False).tolist()
                start_time = time.time()
                search_result = qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=5,
                    search_params=SearchParams(hnsw_ef=128),
                    with_payload=True
                )
                end_time = time.time()
                total_time += end_time - start_time
                count += 1

                relevant_context = []
                for i, hit in enumerate(search_result):
                    payload = hit.payload or {}
                    relevant_context.append({
                        "text": payload.get("text", ""),
                        "location": payload.get("country", "N/A"),
                        "filename": payload.get("file", "N/A"),
                        "score": hit.score
                    })

                context_texts = [c['text'] for c in relevant_context]
                context_sets_per_query[subquery] = context_texts

                if relevant_context:
                    combined_score = sum([c['score'] for c in relevant_context]) / len(relevant_context)
                    scores.append(combined_score)

            # Inter-query similarity
            if len(query_group) >= 2 and query_group[0] in context_sets_per_query:
                reference = context_sets_per_query[query_group[0]]
                for subquery in query_group[1:]:
                    if subquery in context_sets_per_query:
                        score = compute_average_similarity(reference, context_sets_per_query[subquery], model)
                        inter_scores.append(score)

        # Hallucination queries
        for h_query in hallucination_queries:
            query_vector = model.encode(h_query, show_progress_bar=False).tolist()
            start_time = time.time()
            hallucinated_context = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=5,
                search_params=SearchParams(hnsw_ef=128),
                with_payload=True
            )
            end_time = time.time()
            total_time += end_time - start_time
            count += 1

            hallucinated_results = [
                {
                    "text": hit.payload.get("text", ""),
                    "score": hit.score
                } for hit in hallucinated_context
            ]

            if hallucinated_results:
                hallucination_score = sum([r['score'] for r in hallucinated_results]) / len(hallucinated_results)
                hallucination_scores.append(hallucination_score)

    avg_score = sum(scores) / len(scores) if scores else 0
    avg_inter_score = sum(inter_scores) / len(inter_scores) if inter_scores else 0
    avg_hallucination = sum(hallucination_scores) / len(hallucination_scores) if hallucination_scores else 0
    avg_time_per_query = total_time / count if count > 0 else 0

    return {
        "Model": embedding_model_name,
        "Vector DB": "qdrant_in_memory",
        "score": round(avg_score, 4),
        "Similarity": round(avg_inter_score, 4),
        "Hallucination": round(avg_hallucination, 4),
        "Time (s)": round(avg_time_per_query, 4)
    }


def main():
    client = QdrantClient(":memory:")
    results = []
    for model_name in model_names:
        try:
            result = evaluate_model(model_name, client)
            results.append(result)
        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")

    df = pd.DataFrame(results)
    df.to_csv("qdrant_similarity_results.csv", index=False)
    print(df)


if __name__ == "__main__":
    main()