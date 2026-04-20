from src.retrieval import retrieve_top_k
from src.generation import generate_answer, get_llm, call_gemini
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate_query_variants(query: str, num_variants: int = 3) -> list[str]:
    """
    Generate multiple search query variants using LLM.
    """
    prompt_template = ChatPromptTemplate.from_template("""Generate {num_variants} different search query variants for the given user question.
These variants should help capture different aspects of the information needed.
One query per line. Do not number them.

User Question: {query}""")
    
    chain = prompt_template | get_llm() | StrOutputParser()
    
    try:
        response_text = chain.invoke({"num_variants": num_variants, "query": query})
        variants = [v.strip() for v in response_text.split("\n") if v.strip()]
        return variants[:num_variants]
    except Exception as e:
        print(f"Error generating variants: {e}")
        return [query]

def reciprocal_rank_fusion(results_list: list[list[dict]], k: int = 60):
    """
    Reciprocal Rank Fusion (RRF) to merge multiple ranked lists.
    """
    fused_scores = {}
    for results in results_list:
        for rank, res in enumerate(results):
            doc_text = res["text"]
            if doc_text not in fused_scores:
                fused_scores[doc_text] = {"score": 0.0, "metadata": res["metadata"]}
            fused_scores[doc_text]["score"] += 1.0 / (rank + k)
            
    # Sort by fused score descending
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1]["score"], reverse=True)
    
    # Return in standard format
    return [{"text": text, "score": val["score"], "metadata": val["metadata"]} for text, val in sorted_docs]

def run_rag_fusion(query: str, top_k: int = 5):
    """
    Execute RAG Fusion pipeline.
    """
    # 1. Generate variants
    variants = generate_query_variants(query) or [query]
    if query not in variants:
        variants.append(query)
    
    # 2. Retrieve for each
    all_results = []
    for q in variants:
        all_results.append(retrieve_top_k(q, top_k=top_k*2)) # Retrieve extra for better fusion
        
    # 3. Fuse
    fused_results = reciprocal_rank_fusion(all_results)
    final_context = fused_results[:top_k]
    
    # 4. Generate
    answer = generate_answer(query, final_context)
    return answer, final_context

if __name__ == "__main__":
    # Test
    ans, context = run_rag_fusion("Who won the 2024 S&P 500 performance battle?")
    print(f"Answer: {ans}")
