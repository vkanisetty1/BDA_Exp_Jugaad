from langchain_community.embeddings import HuggingFaceEmbeddings
import time


def create_embeddings(text_chunks: list) -> list:
    """
    Generate vector embeddings for text chunks.

    Args:
        text_chunks (list): List of text chunks.

    Returns:
        list: Vector embeddings for the text chunks.
    """
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #embeddings_model = HuggingFaceEmbeddings()
    # this line is taking too much time 
   
    start_time = time.time()
    embeddings = [embeddings_model.embed_query(chunk) for chunk in text_chunks]
    print(f"Time taken: {time.time() - start_time} seconds")
    return embeddings


#embeddings = [embeddings_model.embed_query(chunk) for chunk in text_chunks]


