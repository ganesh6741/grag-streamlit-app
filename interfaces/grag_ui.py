# grag_ui.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from grag_modules.prompter import build_prompt
from grag_modules.retriever import search_chunks
from grag_modules.llm_wrapper import query_model

st.set_page_config(page_title="GRAG Interface", layout="wide")
st.title("📄 GRAG: Academic Insight Generator")

query = st.text_input("Ask a question about research:")
if query:
    chunks = search_chunks(query)
    st.write(f"Found {len(chunks)} relevant chunks.")

    for i, chunk in enumerate(chunks):
        with st.expander(f"Chunk {i+1}"):
            st.write(chunk)
            prompt = build_prompt("qa", chunk, query)
            reply = query_model(prompt)
            st.markdown(f"**Answer:** {reply}")