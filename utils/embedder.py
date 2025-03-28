import tiktoken
import numpy as np

def chunk_text(text, max_tokens=500, overlap=100):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk = tokenizer.decode(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks

def embed_chunks(chunks, client):
    vectors = []
    progress = 0
    with_progress = True

    if with_progress:
        import streamlit as st
        bar = st.progress(0, text="🔄 Embedding chunks...")

    for i, chunk in enumerate(chunks):
        response = client.embeddings.create(
            input=chunk,
            model="text-embedding-ada-002"
        )
        vectors.append(response.data[0].embedding)

        if with_progress:
            bar.progress((i + 1) / len(chunks), text=f"🔄 Embedding chunk {i+1}/{len(chunks)}")

    if with_progress:
        bar.empty()

    return vectors
