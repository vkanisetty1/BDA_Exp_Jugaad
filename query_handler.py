from embedding_generator import create_embeddings
from vector_store import retrieve_from_store
from llm_integration import generate_response

def handle_query(query: str, store_path: str) -> str:
    """
    Process a user query and return a generated response.

    Args:
        query (str): User's query.
        store_path (str): Path to the FAISS vector store.

    Returns:
        str: Response from the chatbot.
    """
    query_embedding = create_embeddings([query])[0]
    relevant_texts = retrieve_from_store(query_embedding, store_path)
    context = " ".join([text["text"] for text in relevant_texts])
    return generate_response(context + "\n" + query)
