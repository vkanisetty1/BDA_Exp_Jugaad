import streamlit as st
from query_handler import handle_query

def main():
    """
    Main function for the Streamlit frontend.
    """
    st.title("JugaadDoc - Medical Chatbot")
    query = st.text_input("Enter your health-related question:")
    if query:
        store_path = "path/to/faiss_index"
        result = handle_query(query, store_path)
        st.write("Response:")
        st.write(result)

if __name__ == "__main__":
    # Ensure this script is run via Streamlit
    main()

