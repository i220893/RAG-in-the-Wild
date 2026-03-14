import os
import yaml
from src.data_loader import load_examples
from src.pipelines import PIPELINES
from src.evaluation import evaluate_accuracy
from src.corpus import build_index
from tqdm import tqdm
import time

def run_evaluation(limit: int = 50):
    print(f"Starting evaluation on up to {limit} examples...")
    
    # 1. Ensure index is ready
    # Note: build_index checks if data already exists in chroma
    build_index()
    
    stats = {name: {"correct": 0, "total": 0, "times": []} for name in PIPELINES}
    
    examples = list(load_examples(limit=limit))
    
    for ex in tqdm(examples):
        query = ex["query"]
        gold_ans = ex["answer"]
        alt_ans = ex["alt_ans"]
        
        for name, run_fn in PIPELINES.items():
            start_time = time.time()
            try:
                pred_ans, _ = run_fn(query)
                correct = evaluate_accuracy(pred_ans, gold_ans, alt_ans)
                if correct:
                    stats[name]["correct"] += 1
            except Exception as e:
                print(f"Error evaluating {name} on query '{query}': {e}")
            
            stats[name]["total"] += 1
            stats[name]["times"].append(time.time() - start_time)
            
    # Report results
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"{'Pipeline':<15} | {'Accuracy':<10} | {'Avg Time (s)':<12}")
    print("-" * 43)
    
    for name, data in stats.items():
        acc = (data["correct"] / data["total"]) * 100 if data["total"] > 0 else 0
        avg_t = sum(data["times"]) / len(data["times"]) if data["times"] else 0
        print(f"{name:<15} | {acc:>8.2f}% | {avg_t:>12.2f}")
    print("="*40)

if __name__ == "__main__":
    # You might want to run with a smaller limit first
    run_evaluation(limit=5)
