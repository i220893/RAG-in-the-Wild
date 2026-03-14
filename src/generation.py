import google.generativeai as genai
import yaml
from pathlib import Path
import time

# Load config
def load_config():
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        config_path = Path("config/config.example.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# Configure Gemini
api_key = CONFIG.get("google_api_key")
if api_key:
    genai.configure(api_key=api_key)
else:
    import os
    env_key = os.environ.get("GOOGLE_API_KEY")
    if env_key:
        genai.configure(api_key=env_key)

def call_gemini(prompt: str, model_id: str = None) -> str:
    """
    Central wrapper for Gemini calls with retry logic.
    """
    mid = model_id or CONFIG.get("generation_model", "gemini-1.5-flash") # Fallback to 1.5 if 2.0 is overloaded
    max_retries = 5
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(mid)
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                if attempt < max_retries - 1:
                    delay = (attempt + 1) * 60 # Wait 60, 120, 180...
                    print(f"Rate limited (429). Attempt {attempt+1}. Waiting {delay}s...")
                    time.sleep(delay)
                    continue
            raise e

def generate_answer(query: str, context_chunks: list[dict], model_name: str = None) -> str:
    # Construct context string
    context_str = ""
    for i, chunk in enumerate(context_chunks):
        context_str += f"[Source {i+1}]: {chunk['text']}\n\n"
    
    prompt = f"""You are a helpful assistant. Answer the user question accurately based ONLY on the provided context artifacts.
If the context doesn't contain the answer, say that you don't know based on the search results.

Context:
{context_str}

User Question: {query}

Answer:"""

    try:
        return call_gemini(prompt, model_name)
    except Exception as e:
        return f"Error during generation: {str(e)}"

if __name__ == "__main__":
    # Test
    print("Testing generation...")
    print(generate_answer("What is the capital of France?", [{"text": "Paris is the capital."}]))
