from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import logging

# Logger setup
logger = logging.getLogger(__name__)

# Global variables to cache models
llm_pipeline = None
llama_tokenizer = None
llama_model = None


def get_device():
    """
    Detect the device to use for the pipeline.
    
    Returns:
        int: 0 for GPU, -1 for CPU.
    """
    if torch.cuda.is_available():
        logger.info("GPU detected. Using GPU for the pipeline.")
        return 0  # Use the first GPU
    else:
        logger.info("No GPU detected. Using CPU for the pipeline.")
        return -1  # Use CPU
    

def get_llm_pipeline():
    """
    Initialize and cache the HuggingFace pipeline for Llama 2.

    Returns:
        transformers.pipeline: A text generation pipeline.
    """
    global llm_pipeline, llama_tokenizer, llama_model
    if llm_pipeline is None:
        logger.info("Loading cached Llama-2 model...")
        # Use cached tokenizer and model
        llama_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
        llama_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf')

        device = get_device()
        
        # Create a Hugging Face pipeline for text generation
        llm_pipeline = pipeline(
            task="text-generation",
            model=llama_model,
            tokenizer=llama_tokenizer,
            device=device  # Dynamically detected device # Use GPU if available; use -1 for CPU
        )
        logger.info("Llama-2 pipeline loaded successfully.")
    return llm_pipeline

def generate_response(prompt: str) -> str:
    """
    Generate a response using the cached LLM pipeline.

    Args:
        prompt (str): Input prompt for the model.

    Returns:
        str: Generated response from the model.
    """
    # Retrieve the LLM pipeline instance
    llm_pipeline = get_llm_pipeline()

    # Generate response
    logger.debug(f"Generating response for prompt: {prompt}")
    outputs = llm_pipeline(prompt, max_length=64, temperature=0.7, do_sample=True)
    response = outputs[0]["generated_text"]
    logger.debug(f"Generated response: {response}")
    return response
