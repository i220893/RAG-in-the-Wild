import networkx as nx
from src.retrieval import retrieve_top_k, get_index
from src.generation import generate_answer

def run_graph_rag(query: str, top_k: int = 5):
    """
    Execute Graph RAG pipeline.
    """
    # 1. Standard vector retrieval for initial seeds
    seed_context = retrieve_top_k(query, top_k=top_k)
    
    # 2. Graph Augmentation
    # Strategy: Find additional snippets from the same 'interaction_id' or 'page_url'
    # as the top results to provide broader context.
    augmented_context = list(seed_context)
    seen_texts = set(c['text'] for c in seed_context)
    
    # Simple expansion: retrieve more chunks and filter for source overlap
    # This simulates a "graph-aware" retrieval over the same corpus metadata.
    # In LangChain, we would typically use a SelfQueryRetriever or a MetadataRetriever,
    # but here we follow the "graph-augmented" logic on existing corpus.
    expansion = retrieve_top_k(query, top_k=top_k * 3)
    
    # Link expansion: prioritize chunks that share metadata (interaction_id or page_url) with seeds
    seed_urls = set(c['metadata'].get('page_url') for c in seed_context if c['metadata'].get('page_url'))
    
    for item in expansion:
        if item['text'] in seen_texts:
            continue
        
        item_url = item['metadata'].get('page_url')
        if item_url in seed_urls:
            # Found a "related" node in the graph of sources
            augmented_context.append(item)
            seen_texts.add(item['text'])
            
        if len(augmented_context) >= top_k + 2: # Add a few neighbors
            break
            
    # 3. Generate
    answer = generate_answer(query, augmented_context[:top_k+2])
    return answer, augmented_context[:top_k+2]

if __name__ == "__main__":
    # Test
    ans, context = run_graph_rag("Who directed Inception?")
    print(f"Answer: {ans}")
