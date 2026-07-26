import sys
sys.path.insert(0, '.')
import pandas as pd
from src.models.llm_baseline import predict_llm_baseline, _call_ollama

print("Warming up phi-4...")
_call_ollama("warmup")
print("Warmup done. Starting eval...")

df = pd.read_parquet('data_splits/llm_test_subset.parquet')
results = predict_llm_baseline(df['text'], verbose=True)
results['true_label'] = df['label'].values
results.to_csv('results/llm_baseline_results.csv', index=False)
print(f"Done. Saved {len(results)} rows to results/llm_baseline_results.csv")
print(f"Failed calls: {(results['document_label'] == -1).sum()}")
