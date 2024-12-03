import PyPDF2

def load_data(file_path: str) -> str:
    """
    Load text data from a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    with open(file_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def preprocess_text(text: str) -> list:
    """
    Preprocess the text: remove stopwords, apply stemming, etc.

    Args:
        text (str): Raw text data.

    Returns:
        list: Preprocessed text chunks.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)
