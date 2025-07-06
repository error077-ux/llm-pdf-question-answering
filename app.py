import streamlit as st
import os
import faiss
import numpy as np

# Import functions from your utility modules
from pdf_utils import extract_text_from_pdf
from embed_utils import chunk_text, generate_embeddings
from vector_store import create_faiss_index, search_faiss_index
from llm_utils import load_llm, generate_answer_from_context # Make sure LLM_MODEL_FILENAME is set correctly in llm_utils.py!

# --- Constants and Global Variables ---
# Path to store processed data and FAISS index
PROCESSED_DATA_DIR = "data"
FAISS_INDEX_PATH = os.path.join(PROCESSED_DATA_DIR, "faiss_index.bin")
CHUNK_TEXT_PATH = os.path.join(PROCESSED_DATA_DIR, "chunks.txt")

# Ensure the data directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# --- Streamlit UI Configuration ---
st.set_page_config(layout="wide", page_title="Local LLM PDF Q&A")

st.title("📄 Local LLM-Powered PDF Q&A System")
st.markdown("""
Upload a PDF document and ask questions about its content.
The system uses a local Large Language Model (LLM) to answer your questions,
ensuring your data remains private and responses are based *only* on the uploaded document.
""")

# --- Session State Management ---
# Use Streamlit's session state to store variables across reruns
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'pdf_text_chunks' not in st.session_state:
    st.session_state.pdf_text_chunks = []
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'llm_loaded' not in st.session_state:
    st.session_state.llm_loaded = False
if 'llm_instance' not in st.session_state:
    st.session_state.llm_instance = None


# --- Load LLM (once) ---
@st.cache_resource # Use Streamlit's caching to load the LLM only once
def get_llm():
    if not st.session_state.llm_loaded:
        with st.spinner("🚀 Loading Local LLM (this may take a few moments)..."):
            st.session_state.llm_instance = load_llm()
            if st.session_state.llm_instance:
                st.session_state.llm_loaded = True
                st.success("Local LLM loaded successfully!")
            else:
                st.error("Failed to load LLM. Check `llm_utils.py` and model path.")
    return st.session_state.llm_instance

llm = get_llm()

# --- PDF Upload and Processing ---
st.header("Upload PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Check if a new file is uploaded or if the session state is reset
    if st.session_state.get('last_uploaded_file_name') != uploaded_file.name:
        st.session_state.pdf_processed = False # Mark for re-processing
        st.session_state.last_uploaded_file_name = uploaded_file.name
        st.session_state.pdf_text_chunks = []
        st.session_state.faiss_index = None

    if not st.session_state.pdf_processed:
        with st.spinner("Processing PDF... This may take a while for large documents."):
            # Save the uploaded file temporarily
            pdf_path = os.path.join(PROCESSED_DATA_DIR, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 1. Extract Text
            st.info("Extracting text from PDF...")
            full_text = extract_text_from_pdf(pdf_path)

            if full_text:
                # 2. Chunk Text
                st.info("Chunking text...")
                chunks = chunk_text(full_text)
                st.session_state.pdf_text_chunks = chunks
                st.success(f"Extracted and chunked {len(chunks)} text chunks.")

                # Save chunks for debugging/inspection (optional)
                with open(CHUNK_TEXT_PATH, "w", encoding="utf-8") as f:
                    for i, chunk in enumerate(chunks):
                        f.write(f"--- Chunk {i+1} ---\n")
                        f.write(chunk)
                        f.write("\n\n")

                # 3. Generate Embeddings
                st.info("Generating embeddings for chunks...")
                if len(chunks) > 0:
                    embeddings = generate_embeddings(chunks)

                    # 4. Create FAISS Index
                    st.info("Creating FAISS index...")
                    faiss_index = create_faiss_index(embeddings)
                    st.session_state.faiss_index = faiss_index
                    st.success("FAISS index created successfully!")

                    # Optionally save FAISS index to disk (for persistence)
                    # faiss.write_index(faiss_index, FAISS_INDEX_PATH)
                    # st.info(f"FAISS index saved to {FAISS_INDEX_PATH}")

                    st.session_state.pdf_processed = True
                    st.success("PDF processing complete! You can now ask questions.")
                else:
                    st.warning("No text extracted from PDF. Please try another file.")
                    st.session_state.pdf_processed = False
            else:
                st.error("Failed to extract text from PDF. Please check the file.")
                st.session_state.pdf_processed = False

# --- Q&A Section ---
if st.session_state.pdf_processed and st.session_state.llm_loaded:
    st.header("Ask a Question")
    question = st.text_input("Enter your question here:")

    if question:
        if st.session_state.faiss_index is not None and st.session_state.pdf_text_chunks:
            with st.spinner("Searching for relevant information and generating answer..."):
                # 1. Generate embedding for the question
                query_embedding = generate_embeddings(question)

                # 2. Search FAISS index for top-K similar chunks
                # You can adjust k (number of chunks to retrieve)
                distances, indices = search_faiss_index(st.session_state.faiss_index, query_embedding, k=5)

                # Ensure indices are valid and not -1 (FAISS returns -1 for not found)
                valid_indices = [idx for idx in indices[0] if idx != -1 and idx < len(st.session_state.pdf_text_chunks)]

                if valid_indices:
                    # Retrieve the actual text chunks
                    retrieved_chunks = [st.session_state.pdf_text_chunks[idx] for idx in valid_indices]

                    # 3. Generate answer using LLM with context
                    answer = generate_answer_from_context(question, retrieved_chunks)
                    st.subheader("Answer:")
                    st.write(answer)

                    with st.expander("See Retrieved Context Chunks"):
                        for i, chunk in enumerate(retrieved_chunks):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.write(chunk)
                            st.markdown("---")
                else:
                    st.warning("Could not find relevant information in the uploaded PDF for your question.")
        else:
            st.error("PDF processing incomplete or FAISS index not available.")
elif uploaded_file is None:
    st.info("Upload a PDF to begin the Q&A process.")
elif not st.session_state.llm_loaded:
    st.warning("Waiting for LLM to load...")