# Cleaned RAG system script with consistent article metadata handling
# Full version with indexing, retrieval, OpenAI analysis, and clean metadata tracking

import os
import re
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import hashlib
import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import MarkdownHeaderTextSplitter
import chromadb
from chromadb.config import Settings
from geolocation import get_location_nominatim

# --- Configuration ---
load_dotenv()

# OpenAI Configuration
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# address = "13 Parkgate St, Stoneybatter"
# target_location = get_location_nominatim(address)
target_location = {'city': 'None', 'province': 'None', 'country': 'Japan'}

# --- Utility Functions ---
def get_file_hash(filepath):
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

def extract_metadata_from_path(file_path: str, text: str) -> Dict[str, str]:
    location = os.path.basename(os.path.dirname(file_path)) or "Unknown"
    date_match = re.search(r"(?:operation on|amendments up to and including)\s+(\d{1,2} \w+ \d{4})", text)
    try:
        date_parsed = datetime.strptime(date_match.group(1), "%d %B %Y").date().isoformat() if date_match else None
    except ValueError:
        date_parsed = None
    title_match = re.search(r"(BUILDING CONTROL ACT \d{4}(?:.*?)?)\n", text, re.IGNORECASE)
    description = title_match.group(1).strip() if title_match else os.path.basename(file_path)
    return {"location": location, "date_iso": date_parsed, "description": description}

def clean_metadata(metadata: Dict[str, Optional[str]]) -> Dict[str, str]:
    return {k: (str(v) if v is not None else "unknown") for k, v in metadata.items()}

# def convert_legal_headers_to_markdown(text: str) -> str:
#     patterns = [
#         r"Artikel\s+(\d+[a-zA-Z]?)\.?",
#         r"Article\s+(\d+[a-zA-Z]?)\.?",
#         r"Art\.?\s+(\d+[a-zA-Z]?)\.?",
#         r"Artigo\s+(\d+[a-zA-Z]?)\.?",
#         r"Artículo\s+(?P<num>\d+[a-zA-Z]?|(?:[a-záéíóúñ]{3,}))\.?",
#         r"§\s*(\d+[a-zA-Z]?)\.?"
#     ]
#     for pattern in patterns:
#         text = re.sub(rf"(?m)^{pattern}", lambda m: f"# {m.group(0)}", text)
#     return text
def convert_legal_headers_to_markdown(text: str) -> str:
    """Convert legal document headers to markdown format."""
    patterns = [
        r"(?m)^Artikel\s+\d+[a-zA-Z]?\.?",
        r"(?m)^Article\s+\d+[a-zA-Z]?\.?",
        r"(?m)^Art\.?\s+\d+[a-zA-Z]?\.?",
        r"(?m)^Artigo\s+\d+[a-zA-Z]?\.?",
        r"(?m)^Artículo\s+(?:\d+[a-zA-Z]?|(?:[a-záéíóúñ]{3,}))\.?",
        r"(?m)^§\s*\d+[a-zA-Z]?\.?"
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, lambda m: f"# {m.group(0).strip()}", text)
    return text

def chunk_section(section_text: str, header: str, chunk_size=500, overlap=100):
    content = section_text.strip()
    sub_chunks = []
    start = 0
    while start < len(content):
        end = start + chunk_size
        body = content[start:end]
        sub_chunks.append(f"{body.strip()}")
        start += chunk_size - overlap
    return sub_chunks

def merge_short_chunks(chunks, min_length=200):
    merged = []
    buffer = ""
    current_article = "Unknown"
    for chunk in chunks:
        text = chunk["text"]
        article = chunk.get("metadata", {}).get("article", "Unknown")
        if len(buffer) + len(text) < min_length:
            buffer += "\n" + text
            current_article = article
        else:
            if buffer:
                merged.append({"text": buffer.strip(), "metadata": {"article": current_article}})
                buffer = ""
            merged.append({"text": text.strip(), "metadata": {"article": article}})
    if buffer:
        merged.append({"text": buffer.strip(), "metadata": {"article": current_article}})
    return merged

def markdown_with_subchunks(text: str, chunk_size=500, overlap=100, min_length=300):
    markdown_text = convert_legal_headers_to_markdown(text)
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "article")], strip_headers=True)
    sections = splitter.split_text(markdown_text)
    all_chunks = []
    for doc in sections:
        article = doc.metadata.get("article", "Unknown")
        content = doc.page_content.strip()
        sub_chunks = chunk_section(content, header=article, chunk_size=chunk_size, overlap=overlap)
        for c in sub_chunks:
            all_chunks.append({"text": c, "metadata": {"article": article}})
    return merge_short_chunks(all_chunks, min_length=min_length)

def setup_chroma_collection():
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"No existing collection to delete: {e}")
    return chroma_client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

def retrieve_relevant_context(query: str, top_k: int = 5) -> List[Dict]:
    embedding = embedding_model.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    context = []
    for i in range(len(results['ids'][0])):
        metadata = results['metadatas'][0][i]
        similarity = 1 - results['distances'][0][i]
        context.append({
            "score": similarity,
            "text": results['documents'][0][i],
            "article": metadata.get("article", "Unknown"),
            "original_filename": metadata.get("original_filename", "Unknown"),
            "location": metadata.get("location", "Unknown"),
            "chunk_index": metadata.get("chunk_index", i)
        })
    with open("merged_chunk.txt", "w", encoding="utf-8") as f:
        f.write(f"Query: {query}\n\n")
        for i, chunk in enumerate(context):
            f.write(f"--- Chunk {i+1} ---\n")
            f.write(f"Score: {chunk['score']:.4f}\n")
            f.write(f"Article: {chunk['article']}\n")
            f.write(f"File: {chunk['original_filename']}\n\n")
            f.write(chunk['text'] + "\n\n")
    return context

def index_data_from_folder(folder_path: str, force_reindex: bool = False):
    if not os.path.isdir(folder_path): return
    existing_docs = collection.get()
    existing_files = {
        meta['original_filename']: {
            'hash': meta.get('file_hash', ''),
            'ids': [existing_docs['ids'][i]]
        }
        for i, meta in enumerate(existing_docs.get('metadatas', []))
        if 'original_filename' in meta
    }
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".txt"): continue
        path = os.path.join(folder_path, filename)
        file_hash = get_file_hash(path)
        if filename in existing_files and not force_reindex and existing_files[filename]['hash'] == file_hash:
            print(f"Skipping unchanged file: {filename}")
            continue
        if filename in existing_files:
            collection.delete(ids=existing_files[filename]['ids'])
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        if not text.strip(): continue
        metadata = parse_filename(filename) or extract_metadata_from_path(path, text)
        chunks = markdown_with_subchunks(text)
        docs = [chunk['text'] for chunk in chunks]
        embs = embedding_model.encode(docs).tolist()
        ids, metadatas = [], []
        for i, chunk in enumerate(chunks):
            ids.append(str(uuid.uuid4()))
            metadatas.append(clean_metadata({
                "original_filename": filename,
                "file_hash": file_hash,
                "chunk_index": i,
                **metadata,
                **chunk["metadata"]
            }))
        collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metadatas)
    print("Indexing complete.")

def clean_index():
    """Remove all documents from the collection"""
    ids = collection.get()['ids']
    if ids:
        collection.delete(ids=ids)
    print("Index cleared")

def get_regulatory_assessment(query: str, context: List[Dict]) -> Tuple[Optional[int], Optional[str]]:
    """Sends context to OpenAI to get a regulatory pressure score and explanation."""
    print("\n--- Generating assessment with OpenAI ---")
    if not context:
        print("No context provided to OpenAI.")
        return None, "No relevant context found in the database to assess the query."

    context_string = ""
    sources = set()
    articles_used = []

    for i, item in enumerate(context):
        article = item.get('article', 'Unknown')
        filename = item.get('original_filename', 'Unknown')
        location = item.get('location', 'Unknown')

        source_info = f"Source {i+1}: [Article: {article}, Location: {location}, File: {filename}]"
        context_string += f"{source_info}\nContent: {item['text']}\n\n"

        if article not in articles_used and article.lower() != "unknown":
            articles_used.append(article)

    prompt = f"""
    You are an AI assistant specialized in analyzing real estate regulations. Your task is to assess the level of regulatory pressure strictly based on the provided legislative context.

    **User Query:** {query}

    **Context from Legislative Documents:**
    --- START CONTEXT ---
    {context_string}
    --- END CONTEXT ---

    **Instructions:**
    1. Review the legislative context carefully in relation to the user query: "{query}".
    2. Assess the level of regulatory pressure strictly based on the content of the context. Do not speculate or use outside knowledge.
    3. Focus your reasoning on **specific and countable obligations, protections, or restrictions** found in the text. These may include:
        - Number and severity of constraints on rent increase, development, or eviction
        - Specific mandatory procedures (e.g., written notice, justification requirements)
        - Protections for tenants or buyers (e.g., legal challenge rights, rent caps, minimum lease durations)
        - Compliance burdens (e.g., multiple-step approval processes, penalties, required documentation)

    4. Provide a **Regulatory Pressure Score** from 1 to 10:
        - 1 = Very Low Pressure (few or no restrictions, favorable to landlords/developers)
        - 5 = Moderate Pressure (balanced or standard regulatory obligations)
        - 10 = Very High Pressure (numerous restrictions, strong tenant protections, complex compliance)

    5. Provide a **precise and concrete explanation** (3–6 sentences). In your explanation:
        - **Cite specific provisions** (e.g., "Article 270b allows tenants to challenge rent increases within 30 days.")
        - **Quantify** wherever possible (e.g., "The law imposes 3 procedural requirements for increasing rent, including...")
        - **Avoid generalities**. Do not use vague summaries like "the law provides protections"—explain **what kind, how many, and how strict.**

    6. You may only refer to the following articles, which were provided in the context:
    {', '.join(articles_used)}

    7. In your answer, list exactly which of these articles informed your explanation, including the filenames.

    8. **IMPORTANT:** If the provided context does not meaningfully relate to the user query, state this clearly and assign a score of **N/A**.

    **Example Format:**

    Score: [1–10 or N/A]  
    Explanation: [Clear, grounded explanation with specific citations and numbers]  
    Articles Used:  
    [Every article number used with filename]
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

if __name__ == "__main__":
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
    TEXT_CHUNK_SIZE = 500
    TEXT_CHUNK_OVERLAP = 100
    COUNTRY = re.sub(r'[^a-zA-Z0-9_-]', '_', target_location.get('country', '').strip())
    DATA_FOLDER = os.path.join("..", "Data_txts", COUNTRY)
    FILENAME_PATTERN = re.compile(r"^(?P<location>.+?)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<desc>.*)\.txt$", re.IGNORECASE)
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    vector_size = embedding_model.get_sentence_embedding_dimension()
    COLLECTION_NAME = f"real_estate_regulations_{COUNTRY}_dim{vector_size}"

    chroma_client = chromadb.PersistentClient(path="chroma_db_data")
    collection = chroma_client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    existing_count = collection.count()
    if existing_count == 0:
        print("\nInitial data load into Chroma...")
        index_data_from_folder(DATA_FOLDER)
    else:
        print(f"\nUsing existing Chroma collection with {existing_count} items")
        # Optional: Uncomment to force reindex if needed
        # print("Forcing reindex...")
        # clean_index()
        # index_data_from_folder(DATA_FOLDER, force_reindex=True)

    for query in ["tenant rights and restrictions"]: 
        relevant_context = retrieve_relevant_context(query)

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

        else:
            print("\n--- Final Assessment ---")
            print(f"Query: {query}")
            print("Could not retrieve relevant context from the database to perform an assessment.")

        print("\n" + "="*50)
        print("Script finished. Chroma data persists for future runs.")
