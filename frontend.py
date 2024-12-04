import streamlit as st
from vector_store import retrieve_from_store
import config
from query_handler import handle_query
#import streamlit as st
#Import logging

vector_store = None  # Declare global variable

def get_vector_store():
    global vector_store
    if vector_store is None:
        vector_store = retrieve_from_store(store_path=config.FAISS_INDEX_PATH)
    return vector_store
@st.cache_data
def load_vector_store():
    from vector_store import retrieve_from_store
    return retrieve_from_store

def main():
    """
    Main function for the Streamlit frontend.
    """
    st.title("JugaadDoc - Medical Chatbot")
    query = st.text_input("Enter your health-related question:")
    if query:
        store_path = config.FAISS_INDEX_PATH
        with st.spinner('Processing your query...'):
         result = handle_query(query, store_path)
        st.success('Done!')
        st.write("Response:")
        st.write(result)

if __name__ == "__main__":
    # Ensure this script is run via Streamlit
    main()

