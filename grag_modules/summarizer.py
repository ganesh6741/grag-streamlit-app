# grag_modules/summarizer.py

from retriever import fetch_chunks_by_id
from prompter import build_prompt
from llm_wrapper import query_model

def summarize_paper(paper_id: str) -> str:
    chunks = fetch_chunks_by_id(paper_id)
    results = []

    for chunk in chunks:
        prompt = build_prompt("summarize", chunk)
        reply = query_model(prompt)
        results.append(reply)

    return "\n\n".join(results)

if __name__ == "__main__":
    output = summarize_paper("2304.05678")  # use any valid ID
    print(output)