# embed_utils.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the Sentence Transformer model once
# This model converts text into numerical vectors (embeddings)
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    print("Please ensure you have an active internet connection or the model is cached.")
    embedding_model = None # Set to None if loading fails

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """
    Splits a given text into smaller, overlapping chunks.

    Args:
        text (str): The input text to chunk.
        chunk_size (int): The maximum size of each chunk in characters.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        list[str]: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def generate_embeddings(texts: list[str]) -> np.ndarray:
    """
    Generates embeddings for a list of text strings using the SentenceTransformer model.

    Args:
        texts (list[str]): A list of text strings (chunks).

    Returns:
        np.ndarray: A NumPy array where each row is an embedding vector for a text chunk.
    """
    if embedding_model is None:
        raise RuntimeError("Embedding model not loaded. Cannot generate embeddings.")
    
    # Encode the texts to get embeddings
    # Using 'batch_size' can speed up processing for many texts
    embeddings = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings

if __name__ == '__main__':
    # Example usage:
    sample_text = "This is a long sample text to demonstrate text chunking. It needs to be long enough to be split into multiple pieces. The chunks will have some overlap to maintain context across boundaries. This helps the LLM to get full context during retrieval."
    
    print("Original Text Length:", len(sample_text))
    
    chunks = chunk_text(sample_text, chunk_size=50, chunk_overlap=10)
    print("\n--- Chunks Generated ---")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} (Length: {len(chunk)}): {chunk[:50]}...") # Print first 50 chars
    
    if embedding_model:
        try:
            embeddings = generate_embeddings(chunks)
            print(f"\nGenerated {len(embeddings)} embeddings, each with dimension {embeddings.shape[1]}.")
            # print(f"First embedding (first 5 values): {embeddings[0][:5]}")
        except RuntimeError as e:
            print(e)
    else:
        print("\nSkipping embedding generation as model failed to load.")