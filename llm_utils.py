# llm_utils.py
import os
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Configuration for your LLM ---
# Set the correct filename of your downloaded GGUF model
# Example: "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"
LLM_MODEL_FILENAME = "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf" # <--- **UPDATE THIS LINE**
LLM_MODEL_PATH = os.path.join("models", LLM_MODEL_FILENAME)

def load_llm():
    """
    Loads the local LlamaCpp LLM model.
    """
    if not os.path.exists(LLM_MODEL_PATH):
        print(f"Error: LLM model not found at {LLM_MODEL_PATH}")
        print("Please ensure your LLM model is downloaded and placed in the 'models/' directory and LLM_MODEL_FILENAME is correct.")
        return None

    try:
        # Initialize LlamaCpp LLM
        # n_gpu_layers=-1 means all layers on GPU if CUDA is available, otherwise CPU
        # n_batch: Number of tokens to process in parallel
        # n_ctx: Context window size (how many tokens the model can "see")
        llm = LlamaCpp(
            model_path=LLM_MODEL_PATH,
            temperature=0.1, # Lower for more factual, higher for more creative
            max_tokens=500,  # Max number of tokens to generate in response
            n_gpu_layers=0,  # Set to 0 for CPU-only, or adjust for GPU (e.g., -1 for all)
            n_batch=512,
            n_ctx=2048,      # Must be larger than combined prompt + generated tokens
            verbose=False,   # Set to True for verbose output from LlamaCpp
            streaming=True   # Enable streaming for better UI experience
        )
        return llm
    except Exception as e:
        print(f"Failed to load LLM model from {LLM_MODEL_PATH}: {e}")
        print("Please check the model path, ensure it's a valid GGUF file, and verify your llama-cpp-python installation.")
        return None

def generate_answer_from_context(question: str, context_chunks: list[str]) -> str:
    """
    Generates an answer to a question based on provided context chunks using the LLM.

    Args:
        question (str): The user's question.
        context_chunks (list[str]): A list of text chunks relevant to the question.

    Returns:
        str: The generated answer from the LLM.
    """
    llm_instance = load_llm()
    if llm_instance is None:
        return "Error: LLM not loaded. Cannot generate answer."

    context = "\n\n".join(context_chunks)

    # Define the prompt template
    # We use a specific instruction format for chat models, e.g., Llama-2 chat
    template = """You are an AI assistant specialized in answering questions based on the provided text context.
Do not make up answers. If the answer cannot be found in the context, clearly state that.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm_instance)

    try:
        response = llm_chain.invoke({"context": context, "question": question})
        # For invoke, response is a dictionary, extract 'text'
        return response.get('text', 'No answer generated.')
    except Exception as e:
        return f"An error occurred during LLM inference: {e}"

if __name__ == '__main__':
    # Example usage:
    # Ensure a valid LLM model path is configured above.
    # This might take a moment to load the model.
    
    print(f"Attempting to load LLM from: {LLM_MODEL_PATH}")
    test_llm = load_llm()
    if test_llm:
        print("LLM loaded successfully!")
        
        test_context = [
            "The quick brown fox jumps over the lazy dog.",
            "Dogs are common pets, known for their loyalty and playfulness. They come in many breeds."
        ]
        test_question = "What is a common characteristic of dogs?"
        
        print(f"\nQuestion: {test_question}")
        print(f"Context: {test_context}")
        
        answer = generate_answer_from_context(test_question, test_context)
        print(f"\nGenerated Answer: {answer}")
    else:
        print("\nCould not load LLM for testing. Check model path and filename.")