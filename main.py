import os
from data_loader import load_data, preprocess_text
from embedding_generator import create_embeddings
from vector_store import save_to_vector_store
import config

def main():
    """
    Main function to preprocess data, generate embeddings, and save to FAISS index.
    """
    # Load and preprocess data
    raw_text = load_data(config.PDF_FILE_PATH)
    text_chunks = preprocess_text(raw_text)

    # Generate embeddings
    embeddings = create_embeddings(text_chunks)

    # Save embeddings to FAISS vector store
    save_to_vector_store(embeddings, text_chunks, config.FAISS_INDEX_PATH)

    # Instructions to launch the Streamlit frontend
    print("To run the Streamlit frontend, use the following command:")
    print("streamlit run frontend.py")

if __name__ == "__main__":
    # Ensure this script focuses on backend processing
    main()


