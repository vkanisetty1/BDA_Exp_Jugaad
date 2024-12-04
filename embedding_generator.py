from langchain_community.embeddings import HuggingFaceEmbeddings
import time


def create_embeddings(text_chunks: list) -> list:
    """
    Generate vector embeddings for text chunks using batch processing.

    Args:
        text_chunks (list): List of text chunks.

    Returns:
        list: Vector embeddings for the text chunks.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    start_time = time.time()
    embeddings = embeddings_model.embed_documents(text_chunks)  # Batch embedding
    print(f"Time taken: {time.time() - start_time} seconds")
    return embeddings


#embeddings = [embeddings_model.embed_query(chunk) for chunk in text_chunks]


