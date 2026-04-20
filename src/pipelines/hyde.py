from src.retrieval import retrieve_top_k
from src.generation import generate_answer, get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate_hypothetical_doc(query: str) -> str:
    """
    Generate a hypothetical document that might contain the answer.
    """
    prompt_template = ChatPromptTemplate.from_template("""Write a single paragraph, factual-sounding hypothetical response that answers the following question.
The goal is to provide a text that looks like a high-quality search snippet or Wikipedia-style paragraph.

Question: {question}""")
    
    chain = prompt_template | get_llm() | StrOutputParser()
    
    try:
        return chain.invoke({"question": query})
    except Exception as e:
        print(f"Error generating hypo doc: {e}")
        return query

def run_hyde(query: str, top_k: int = 5):
    """
    Execute HyDE pipeline.
    """
    # 1. Generate hypothetical doc
    hypo_doc = generate_hypothetical_doc(query)
    
    # 2. Retrieve using hypo doc
    context = retrieve_top_k(hypo_doc, top_k=top_k)
    
    # 3. Generate final answer
    answer = generate_answer(query, context)
    return answer, context

if __name__ == "__main__":
    # Test
    ans, context = run_hyde("Which athlete has won more Grand Slams, Federer or Nadal?")
    print(f"Answer: {ans}")
