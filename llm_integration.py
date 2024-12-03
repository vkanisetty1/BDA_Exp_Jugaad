from langchain_community.llms import HuggingFacePipeline

def generate_response(prompt: str) -> str:
    """
    Generate a response using Llama 2.

    Args:
        prompt (str): Input prompt.

    Returns:
        str: Generated response.
    """
    llm = HuggingFacePipeline.from_model_id(
        model_id="meta-llama/Llama-2-7b-chat-hf",
        task="text-generation",
        model_kwargs={"temperature": 0, "max_length": 64}
    )
    return llm(prompt)
