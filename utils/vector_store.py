# utils/vector_store.py

import faiss
import os
import pickle
import numpy as np

VECTOR_DIR = "data/vector_store"
os.makedirs(VECTOR_DIR, exist_ok=True)

def save_faiss_index(index, doc_name, metadata):
    faiss.write_index(index, os.path.join(VECTOR_DIR, f"{doc_name}.index"))

    with open(os.path.join(VECTOR_DIR, f"{doc_name}_meta.pkl"), "wb") as f:
        pickle.dump(metadata, f)

def load_faiss_index(doc_name):
    index_path = os.path.join(VECTOR_DIR, f"{doc_name}.index")
    meta_path = os.path.join(VECTOR_DIR, f"{doc_name}_meta.pkl")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return None, None

    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata

def search_index(index, query_vector, k=15):
    """Return top k most similar chunks to the query"""
    distances, indices = index.search(np.array([query_vector]).astype("float32"), k)
    return indices[0], distances[0]
