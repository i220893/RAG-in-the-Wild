import yaml
from pathlib import Path
import time
import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load config
def load_config():
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        config_path = Path("config/config.example.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# Initialize LLM
_LLM = None

def get_llm(model_id: str = None):
    global _LLM
    if _LLM is None or model_id is not None:
        mid = model_id or CONFIG.get("generation_model", "qwen2.5:7b-instruct-q4_K_M")
        _LLM = ChatOllama(
            model=mid,
            temperature=0
        )
    return _LLM

def call_gemini(prompt: str, model_id: str = None) -> str:
    """
    Wrapper for LLM calls using LangChain.
    """
    llm = get_llm(model_id)
    response = llm.invoke(prompt)
    return response.content.strip()

def generate_answer(query: str, context_chunks: list[dict], model_name: str = None) -> str:
    # Construct context string
    context_str = ""
    for i, chunk in enumerate(context_chunks):
        context_str += f"[Source {i+1}]: {chunk['text']}\n\n"
    
    prompt_template = ChatPromptTemplate.from_template("""You are a helpful assistant. Answer the user question accurately based ONLY on the provided context artifacts.
Provide ONLY the direct, concise answer without any conversational filler or introductions.
If the context doesn't contain the answer, say "I don't know".

Context:
{context}

User Question: {query}

Answer:""")

    chain = prompt_template | get_llm(model_name) | StrOutputParser()
    
    try:
        return chain.invoke({"context": context_str, "query": query})
    except Exception as e:
        return f"Error during generation: {str(e)}"

if __name__ == "__main__":
    # Test
    print("Testing generation...")
    print(generate_answer("What is the capital of France?", [{"text": "Paris is the capital."}]))
