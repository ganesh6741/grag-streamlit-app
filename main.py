from grag_modules.abstract_summarizer import summarize
from utils.data_loader import load_data, clean_text
from embeddings.embedding_manager import add_to_index, query_index
from prompts.prompt_generator import create_prompt

def run_pipeline():
    # Load and clean abstracts
    abstracts = load_data("data/raw/abstracts.csv")
    cleaned_abstracts = [clean_text(text) for text in abstracts]

    # Add embeddings to index
    add_to_index(cleaned_abstracts)

    # Example use case: summarize a chosen abstract
    sample = cleaned_abstracts[0]
    print("🧾 Original:\n", sample)

    prompt = create_prompt(sample, mode="summary")
    print("🧠 Prompt:\n", prompt)

    summary = summarize(sample)
    print("🧠 Summary:\n", summary)

    # Query similar abstracts
    print("🔍 Similar abstracts by index position:")
    similar = query_index(sample, top_k=3)
    print(similar)

if __name__ == "__main__":
    run_pipeline()