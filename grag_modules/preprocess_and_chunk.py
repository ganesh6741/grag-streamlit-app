# grag_modules/preprocess_and_chunk.py

import json
import os
import re
from unidecode import unidecode
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def clean_text(text):
    # Remove LaTeX artifacts and normalize
    text = unidecode(text)
    text = re.sub(r"\$.*?\$", "", text)  # Remove inline math
    text = re.sub(r"{.*?}", "", text)   # Remove LaTeX brackets
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text, max_tokens=400, overlap=50):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    token_count = 0

    for sentence in sentences:
        words = sentence.split()
        if token_count + len(words) > max_tokens:
            chunks.append(" ".join(current_chunk))
            # start new chunk with overlap
            current_chunk = current_chunk[-overlap:] + words
            token_count = len(current_chunk)
        else:
            current_chunk.extend(words)
            token_count += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def preprocess_file(filename):
    input_path = os.path.join(RAW_DIR, filename)
    output_path = os.path.join(PROCESSED_DIR, f"chunked_{filename}")

    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            paper = json.loads(line)
            cleaned = clean_text(paper["abstract"])
            chunks = chunk_text(cleaned)
            for i, chunk in enumerate(chunks):
                output = {
                    "paper_id": paper["id"],
                    "chunk_id": f"{paper['id']}_chunk_{i}",
                    "text": chunk
                }
                f_out.write(json.dumps(output) + "\n")

    print(f"Saved cleaned chunks to {output_path}")

if __name__ == "__main__":
    preprocess_file("transformers_arxiv.jsonl")