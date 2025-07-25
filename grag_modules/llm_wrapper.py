# grag_modules/llm_wrapper.py

import openai  # or use HuggingFace locally

openai.api_key = "YOUR_API_KEY"

def query_model(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or your preferred model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()