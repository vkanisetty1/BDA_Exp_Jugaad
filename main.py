import os
from data_loader import load_data, preprocess_text
from embedding_generator import create_embeddings
from vector_store import save_to_vector_store
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import config

def cache_models():
    """
    Download and cache required models locally.
    """
    print("Caching SentenceTransformer model...")
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("SentenceTransformer model cached successfully.")

    print("Caching Llama-2 model...")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    print("Llama-2 model cached successfully.")
def main():
    """
    Main function to preprocess data, generate embeddings, and save to FAISS index.
    """
    # Cache models
    cache_models()
    # Load and preprocess data
    print("--Reading text Data--")
    raw_text = load_data(config.PDF_FILE_PATH)
    print("--Text Data read--")
    text_chunks = preprocess_text(raw_text)
    print("--Text Chunks--")

    # Generate embeddings
    embeddings = create_embeddings(text_chunks)
    print("creating vector dtore")
    # Save embeddings to FAISS vector store
    save_to_vector_store(embeddings, text_chunks, config.FAISS_INDEX_PATH)
    print("created vector store")

    # Instructions to launch the Streamlit frontend
    print("To run the Streamlit frontend, use the following command:")
    print("streamlit run frontend.py")

if __name__ == "__main__":
    # Ensure this script focuses on backend processing
    main()


