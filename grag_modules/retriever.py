# grag_modules/retriever.py

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/vector_store/index.faiss"
ID_MAP_PATH = "data/vector_store/id_map.json"
CHUNK_PATH = "data/processed/chunked_transformers_arxiv.jsonl"

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_PATH)

with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
    ids = json.load(f)

# Load chunks into a lookup table
chunk_lookup = {}
with open(CHUNK_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        chunk_lookup[data["chunk_id"]] = data["text"]

def search_chunks(query: str, top_k: int = 5) -> list:
    query_vec = model.encode(query)
    _, indices = index.search(np.array([query_vec]), top_k)
    return [chunk_lookup[ids[i]] for i in indices[0]]

def fetch_chunks_by_id(paper_id: str) -> list:
    return [chunk for cid, chunk in chunk_lookup.items() if cid.startswith(paper_id)]