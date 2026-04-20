from src.retrieval import retrieve_top_k
from src.generation import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def judge_retrieval_confidence(query: str, context: list[dict]) -> str:
    """
    Assess retrieval confidence (High, Low) using LLM judge.
    """
    context_text = "\n".join([f"- {c['text']}" for c in context])
    
    prompt_template = ChatPromptTemplate.from_template("""Assess whether the following retrieved context is sufficient and relevant to answer the user question.
Rate the confidence as 'HIGH' or 'LOW'.
Output ONLY the word 'HIGH' or 'LOW'.

Question: {query}
Context:
{context}

Confidence:""")

    chain = prompt_template | get_llm() | StrOutputParser()
    
    try:
        decision = chain.invoke({"query": query, "context": context_text}).upper()
        return "HIGH" if "HIGH" in decision else "LOW"
    except Exception as e:
        print(f"Error in CRAG judge: {e}")
        return "HIGH" # Default to high on error

def generate_crag_answer(query: str, context: list[dict], confidence: str) -> str:
    """
    Generate answer with citations. If confidence is low, uses parametric knowledge.
    """
    if confidence == "HIGH":
        context_str = ""
        citations = []
        for i, chunk in enumerate(context):
            source_id = i + 1
            context_str += f"[{source_id}] {chunk['text']}\n\n"
            url = chunk['metadata'].get('page_url', 'N/A')
            citations.append(f"[{source_id}] {url}")
            
        prompt_template = ChatPromptTemplate.from_template("""Answer the user question accurately based on the provided context.
Use citations like [1], [2] in your text where you use information from a source.
If you don't know the answer, say so.

Context:
{context}

User Question: {query}

Answer:""")
        
        chain = prompt_template | get_llm() | StrOutputParser()
        
        try:
            ans_text = chain.invoke({"context": context_str, "query": query})
            citation_footer = "\n\nSources:\n" + "\n".join(citations)
            return ans_text + citation_footer
        except Exception as e:
            return f"Error during CRAG High-Conf generation: {str(e)}"
    else:
        # Low confidence fallback
        prompt_template = ChatPromptTemplate.from_template("""The following question might not be fully answerable from the retrieved web snippets.
Answer the question based on your general knowledge if possible, or explain that the current data is insufficient.

User Question: {query}

Answer:""")
        
        chain = prompt_template | get_llm() | StrOutputParser()
        
        try:
            return chain.invoke({"query": query}) + "\n\n(Note: Retrieval confidence was low; answered from general knowledge.)"
        except Exception as e:
            return f"Error during CRAG Low-Conf generation: {str(e)}"

def run_crag(query: str, top_k: int = 5):
    """
    Execute CRAG pipeline.
    """
    # 1. Retrieve
    context = retrieve_top_k(query, top_k=top_k)
    
    # 2. Judge confidence
    confidence = judge_retrieval_confidence(query, context)
    
    # 3. Generate with citations
    answer = generate_crag_answer(query, context, confidence)
    
    return answer, context

if __name__ == "__main__":
    # Test
    ans, context = run_crag("Who directed Inception?")
    print(f"Answer:\n{ans}")
