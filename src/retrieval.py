from src.corpus import load_index

# Global index instance (singleton for the process)
_GLOBAL_INDEX = None

def get_index():
    global _GLOBAL_INDEX
    if _GLOBAL_INDEX is None:
        _GLOBAL_INDEX = load_index()
    return _GLOBAL_INDEX

def retrieve_top_k(query: str, top_k: int = None):
    """
    Standard retrieval function to be used by all pipelines.
    
    Args:
        query: The user query string.
        top_k: Number of chunks to retrieve (defaults to config value if None).
        
    Returns:
        List of dicts with 'text', 'score', and 'metadata' (including 'page_url').
    """
    index = get_index()
    return index.retrieve(query, top_k=top_k)

if __name__ == "__main__":
    # Quick test
    print("Testing retrieval system...")
    results = retrieve_top_k("What is the capital of France?", top_k=3)
    for i, res in enumerate(results):
        print(f"{i+1}. Score: {res['score']:.4f}")
        print(f"   Text: {res['text'][:100]}...")
