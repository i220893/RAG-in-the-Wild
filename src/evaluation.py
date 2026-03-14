import re

def normalize_answer(s: str) -> str:
    """
    Lowercases, removes punctuation, and extra whitespace for better comparison.
    """
    s = s.lower()
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    s = " ".join(s.split())
    return s

def evaluate_accuracy(prediction: str, gold_answer: str, alt_answers: list[str] = None) -> bool:
    """
    Returns True if prediction matches gold_answer or any of the alt_answers.
    Uses soft matching/substring check.
    """
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold_answer)
    
    # Check exact match or substring
    if gold_norm in pred_norm or pred_norm in gold_norm:
        return True
        
    for alt in (alt_answers or []):
        alt_norm = normalize_answer(alt)
        if alt_norm in pred_norm or pred_norm in alt_norm:
            return True
            
    return False

if __name__ == "__main__":
    # Test
    print(evaluate_accuracy("Christopher Nolan directed Inception.", "Christopher Nolan")) # True
    print(evaluate_accuracy("It was directed by Nolan.", "Christopher Nolan")) # True (nolan match)
    print(evaluate_accuracy("Steven Spielberg.", "Christopher Nolan")) # False
