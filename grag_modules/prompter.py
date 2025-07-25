# grag_modules/prompter.py

import os

PROMPT_DIR = "prompts"

def load_prompt(task: str) -> str:
    path = os.path.join(PROMPT_DIR, f"{task}.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_prompt(task: str, chunk: str, question: str = None) -> str:
    template = load_prompt(task)
    if "{question}" in template and question:
        return template.format(chunk=chunk, question=question)
    else:
        return template.format(chunk=chunk)