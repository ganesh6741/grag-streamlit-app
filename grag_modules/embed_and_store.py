# grag_modules/embed_and_store.py

import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

CHUNK_FILE = "data/processed/chunked_transformers_arxiv.jsonl"
INDEX_FILE = "data/vector_store/index.faiss"
ID_MAP_FILE = "data/vector_store/id_map.json"
os.makedirs("data/vector_store", exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = []
ids = []

with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        text = entry["text"]
        embedding = model.encode(text)
        vectors.append(embedding)
        ids.append(entry["chunk_id"])

index = faiss.IndexFlatL2(len(vectors[0]))
index.add(np.array(vectors))
faiss.write_index(index, INDEX_FILE)

with open(ID_MAP_FILE, "w", encoding="utf-8") as f:
    json.dump(ids, f)

print(f"Saved FAISS index to {INDEX_FILE}")