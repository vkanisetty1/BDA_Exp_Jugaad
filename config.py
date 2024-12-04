PDF_FILE_PATH = "Data/medical_data.pdf"
FAISS_INDEX_PATH = "Data/faiss_index/"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
LLM_MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
