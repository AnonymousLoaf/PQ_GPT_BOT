import streamlit as st
import os
import faiss
import numpy as np
from openai import OpenAI

from utils.file_handler import extract_text
from utils.embedder import chunk_text, embed_chunks
from utils.vector_store import save_faiss_index, load_faiss_index, search_index

# Load .env and OpenAI client
api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

# Streamlit setup
st.set_page_config(page_title="Doc GPT", layout="centered")
st.title("\ud83d\udcc4 Chat with Your Document")

UPLOAD_FOLDER = "data/uploaded_docs"
VECTOR_STORE = "data/vector_store"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE, exist_ok=True)

# Chat state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload doc
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])
file_type = uploaded_file.name.split(".")[-1] if uploaded_file else None

if uploaded_file:
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"\ud83d\udcc1 Saved file to {file_path}")

    # Extract text
    text = extract_text(uploaded_file, file_type)
    st.subheader("\ud83d\udcc4 Extracted Text Preview")
    st.text_area("Document Content", value=text[:3000], height=300)

    # Embed if not already
    index_path = os.path.join(VECTOR_STORE, f"{uploaded_file.name}.index")
    if os.path.exists(index_path):
        st.info("\u2705 Document already embedded. Skipping embedding.")
    else:
        with st.spinner("\ud83d\udd04 Chunking and embedding..."):
            chunks = chunk_text(text)
            vectors = embed_chunks(chunks, client)

            dimension = len(vectors[0])
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(vectors).astype("float32"))

            metadata = {
                "chunks": chunks,
                "file": uploaded_file.name,
            }

            save_faiss_index(index, uploaded_file.name, metadata)
        st.success(f"\u2705 Embedded and saved {len(chunks)} chunks to FAISS index.")

# Divider
st.divider()
st.header("\ud83d\udcac Ask Questions About Your Document")

# Document selector
available_docs = [
    f.replace(".index", "") for f in os.listdir(VECTOR_STORE) if f.endswith(".index")
]
selected_doc = st.selectbox("\ud83d\udcc2 Choose a document to chat with:", options=available_docs) if available_docs else None

# Clear history
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("\ud83e\uddf9 Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.pop("active_question", None)
        st.session_state.pop("active_answer", None)

# Show interactive chat history
if st.session_state.chat_history:
    st.markdown("### \ud83d\udcac Click a question to reload it")
    for i, chat in enumerate(st.session_state.chat_history[::-1]):
        if st.button(f"\ud83d\udde8\ufe0f {chat['question']}", key=f"history_{i}"):
            st.session_state.active_question = chat["question"]
            st.session_state.active_answer = chat["answer"]
            st.rerun()

# Q&A
if selected_doc:
    index, metadata = load_faiss_index(selected_doc)

    if not index or not metadata:
        st.warning("\u26a0\ufe0f Failed to load FAISS index or metadata for this document.")
        st.stop()

    # Show previous selection
    if "active_question" in st.session_state and "active_answer" in st.session_state:
        with st.chat_message("user", avatar="\ud83d\udc64"):
            st.markdown(st.session_state.active_question)
        with st.chat_message("assistant", avatar="\ud83e\udd16"):
            st.markdown(st.session_state.active_answer)

    # New question input
    user_question = st.text_input("Ask a question about the document:")

    if user_question:
        q_embed = client.embeddings.create(
            input=user_question,
            model="text-embedding-ada-002"
        ).data[0].embedding

        indices, distances = search_index(index, q_embed, k=5)
        context_chunks = [metadata["chunks"][i] for i in indices if i < len(metadata["chunks"])]
        context_text = "\n\n---\n\n".join(context_chunks)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Answer the question based only on the document content provided."},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_question}"}
            ]
        )
        answer = response.choices[0].message.content

        with st.chat_message("user", avatar="\ud83d\udc64"):
            st.markdown(user_question)
        with st.chat_message("assistant", avatar="\ud83e\udd16"):
            st.markdown(answer)

        st.session_state.chat_history.append({
            "question": user_question,
            "answer": answer
        })

        st.session_state.active_question = user_question
        st.session_state.active_answer = answer

else:
    st.warning("\ud83d\udcc4 No documents available yet. Please upload and embed one.")
