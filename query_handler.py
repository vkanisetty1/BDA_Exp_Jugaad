from embedding_generator import create_embeddings
from vector_store import retrieve_from_store
from llm_integration import generate_response
import logging

logger = logging.getLogger(__name__)

def handle_query(query: str, store_path: str) -> str:
    logger.debug(f"Processing query: {query}")
    query_embedding = create_embeddings([query])[0]
    logger.debug("Query embedding generated successfully.")
    relevant_texts = retrieve_from_store(query_embedding, store_path)
    logger.debug(f"Relevant texts retrieved: {len(relevant_texts)} items.")
    context = " ".join([text.page_content for text in relevant_texts])
    response = generate_response(context + "\n" + query)
    logger.debug(f"Generated response: {response}")
    return response
