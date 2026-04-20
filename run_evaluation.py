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
    
    with open("evaluation.txt", "w", encoding="utf-8") as f:
        for i, ex in enumerate(tqdm(examples)):
            query = ex["query"]
            gold_ans = ex["answer"]
            alt_ans = ex["alt_ans"]
            
            f.write(f"Example {i+1}\n")
            f.write(f"Query: {query}\n")
            f.write(f"Gold Answer: {gold_ans}\n")
            
            for name, run_fn in PIPELINES.items():
                start_time = time.time()
                try:
                    pred_ans, _ = run_fn(query)
                    correct = evaluate_accuracy(pred_ans, gold_ans, alt_ans)
                    if correct:
                        stats[name]["correct"] += 1
                        
                    f.write(f"\n[{name}]\n")
                    f.write(f"Generated Answer: {pred_ans}\n")
                    f.write(f"LLM Judgement: {'Correct' if correct else 'Incorrect'}\n")
                except Exception as e:
                    print(f"Error evaluating {name} on query '{query}': {e}")
                    f.write(f"\n[{name}]\n")
                    f.write(f"Error: {e}\n")
                
                stats[name]["total"] += 1
                stats[name]["times"].append(time.time() - start_time)
                
            f.write("-" * 80 + "\n\n")
            
        # Report results
        results_str = "\n" + "="*40 + "\n"
        results_str += "EVALUATION RESULTS\n"
        results_str += "="*40 + "\n"
        results_str += f"{'Pipeline':<15} | {'Accuracy':<10} | {'Avg Time (s)':<12}\n"
        results_str += "-" * 43 + "\n"
        
        for name, data in stats.items():
            acc = (data["correct"] / data["total"]) * 100 if data["total"] > 0 else 0
            avg_t = sum(data["times"]) / len(data["times"]) if data["times"] else 0
            results_str += f"{name:<15} | {acc:>8.2f}% | {avg_t:>12.2f}\n"
        results_str += "="*40 + "\n"
        
        print(results_str)
        f.write(results_str)

if __name__ == "__main__":
    # Evaluate a larger, detailed segment (from the first 1000 items that were embedded)
    run_evaluation(limit=20)
