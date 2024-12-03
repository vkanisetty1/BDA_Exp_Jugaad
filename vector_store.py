from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import config


def save_to_vector_store(embeddings: list, texts: list, store_path: str):
    """
    Save embeddings and text chunks to a FAISS vector store.

    Args:
        embeddings (list): List of embeddings.
        texts (list): Corresponding text chunks.
        store_path (str): Path to save the FAISS index.
    """
#     #vector_store = FAISS.from_embeddings(embeddings, texts)
#     #result = FAISS.from_embeddings(embeddings, texts)
#     vector_store, *rest = FAISS.from_embeddings(embeddings, texts)

#    #print(result)
#     vector_store.save_local(store_path)
    # Combine texts and embeddings into the required format
    text_embedding_pairs = list(zip(texts, embeddings))

    # Initialize FAISS vector store from embeddings
    vector_store = FAISS.from_embeddings(text_embedding_pairs, embedding=None)  # Set 'embedding' parameter appropriately if needed
    
    # Save the vector store
    vector_store.save_local(store_path)

def retrieve_from_store(query_embedding, store_path = config.FAISS_INDEX_PATH):
    # Initialize the embeddings model
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load the FAISS vector store with safe deserialization
    vector_store = FAISS.load_local(
        folder_path=store_path,
        embeddings=embeddings_model,
        allow_dangerous_deserialization=True  # Use this only for trusted pickle files
    )
    
    # Perform similarity search
    relevant_texts = vector_store.similarity_search_by_vector(query_embedding, k=5)
    return relevant_texts
