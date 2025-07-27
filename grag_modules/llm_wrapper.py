import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("AIzaSyCGtDbKRJfohhl2fdqNhvMUh-Biqphs2cE"))

model = genai.GenerativeModel("gemini-pro")

def query_model(prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text.strip()
