import arxiv
import json
import os
from tqdm import tqdm

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

def fetch_arxiv_papers(query="natural language processing", max_results=50):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers = []
    for result in tqdm(search.results(), total=max_results, desc=f"Fetching '{query}' papers"):
        paper = {
            "id": result.get_short_id(),
            "title": result.title.strip(),
            "authors": [author.name for author in result.authors],
            "categories": result.categories,
            "abstract": result.summary.strip().replace("\n", " "),
            "published": result.published.isoformat()
        }
        papers.append(paper)

    return papers

def save_jsonl(papers, filename):
    path = os.path.join(RAW_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper) + "\n")
    print(f"Saved {len(papers)} papers to {path}")

if __name__ == "__main__":
    papers = fetch_arxiv_papers("transformers", max_results=100)
    save_jsonl(papers, "transformers_arxiv.jsonl")