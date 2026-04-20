import re

def normalize_answer(s: str) -> str:
    """
    Lowercases, removes punctuation, and extra whitespace for better comparison.
    """
    s = s.lower()
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    s = " ".join(s.split())
    return s

from src.generation import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def evaluate_accuracy(prediction: str, gold_answer: str, alt_answers: list[str] = None) -> bool:
    """
    Returns True if prediction matches gold_answer or any of the alt_answers using an LLM-as-a-judge approach.
    """
    prompt_template = ChatPromptTemplate.from_template("""You are evaluating an AI's predicted answer against the gold standard answer.
Given the Predicted Answer and the Gold Answers, determine if the Predicted Answer correctly provides the core information requested in the Gold Answers.
If the Predicted Answer is essentially correct or contains the right information, return TRUE.
If the Predicted Answer is incorrect, incomplete, or says "I don't know", return FALSE.
Output ONLY 'TRUE' or 'FALSE'.

Gold Answer: {gold_answer}
Alternative Acceptable Answers: {alt_answers}

Predicted Answer: {prediction}

Match (TRUE or FALSE):""")

    chain = prompt_template | get_llm() | StrOutputParser()
    
    try:
        alt_str = ", ".join(alt_answers) if alt_answers else "None"
        result = chain.invoke({
            "gold_answer": gold_answer,
            "alt_answers": alt_str,
            "prediction": prediction
        }).strip().upper()
        
        return "TRUE" in result
    except Exception as e:
        print(f"Error during LLM evaluation: {e}")
        return False

if __name__ == "__main__":
    # Test
    print(evaluate_accuracy("Christopher Nolan directed Inception.", "Christopher Nolan")) # True
    print(evaluate_accuracy("It was directed by Nolan.", "Christopher Nolan")) # True (nolan match)
    print(evaluate_accuracy("Steven Spielberg.", "Christopher Nolan")) # False
