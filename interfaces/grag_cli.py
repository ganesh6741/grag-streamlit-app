# grag_cli.py

from grag_modules.prompter import build_prompt
from grag_modules.retriever import search_chunks
from grag_modules.llm_wrapper import query_model

def cli_loop():
    print("🔍 GRAG CLI Interface")
    while True:
        query = input("\nEnter your question (or 'exit'): ")
        if query.lower() == "exit":
            break

        chunks = search_chunks(query)
        print(f"\nTop {len(chunks)} relevant chunks found.\n")

        for i, chunk in enumerate(chunks):
            prompt = build_prompt("qa", chunk, query)
            reply = query_model(prompt)
            print(f"✅ Result {i+1}:\n{reply}\n")

if __name__ == "__main__":
    cli_loop()