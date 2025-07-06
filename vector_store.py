# vector_store.py
import faiss
import numpy as np

def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Creates a FAISS index from a numpy array of embeddings.

    Args:
        embeddings (np.ndarray): A 2D NumPy array where each row is an embedding vector.

    Returns:
        faiss.IndexFlatL2: A FAISS index that can be searched.
    """
    dimension = embeddings.shape[1]
    # IndexFlatL2 uses Euclidean distance (L2 norm)
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32')) # FAISS expects float32
    return index

def search_faiss_index(index: faiss.IndexFlatL2, query_embedding: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Searches the FAISS index for the top k most similar embeddings to a query embedding.

    Args:
        index (faiss.IndexFlatL2): The FAISS index to search.
        query_embedding (np.ndarray): The embedding vector of the query (1D or 2D array with 1 row).
        k (int): The number of nearest neighbors to retrieve.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - distances (np.ndarray): The distances to the nearest neighbors.
            - indices (np.ndarray): The indices of the nearest neighbors in the original dataset.
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1) # Reshape to 2D for FAISS search

    # FAISS expects float32
    distances, indices = index.search(query_embedding.astype('float32'), k)
    return distances, indices

if __name__ == '__main__':
    # Example usage:
    # Create some dummy embeddings
    dummy_embeddings = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.15, 0.25, 0.35, 0.45],
        [0.9, 0.8, 0.7, 0.6],
        [0.85, 0.75, 0.65, 0.55],
        [0.05, 0.15, 0.25, 0.35]
    ], dtype='float32')

    # Create an index
    faiss_index = create_faiss_index(dummy_embeddings)
    print(f"FAISS index created with {faiss_index.ntotal} vectors.")

    # Create a dummy query embedding
    query_emb = np.array([0.12, 0.22, 0.32, 0.42], dtype='float32')

    # Search the index
    k_value = 2
    distances, indices = search_faiss_index(faiss_index, query_emb, k=k_value)

    print(f"\nTop {k_value} results for query:")
    print("Distances:", distances)
    print("Indices:", indices)

    # Expected: The first two indices (0 and 1) should be closest to the query