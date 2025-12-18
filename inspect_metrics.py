import json
import os
import pandas as pd
import glob

metrics_dir = 'results/metrics'
metric_files = glob.glob(os.path.join(metrics_dir, '**/*.json'), recursive=True)

results = []
for f in metric_files:
    with open(f, 'r') as file:
        results.append(json.load(file))

df = pd.DataFrame(results)
if not df.empty:
    cols = ['model', 'accuracy', 'roc_auc', 'pr_auc', 'brier']
    print(df[cols].sort_values(by='pr_auc', ascending=False).to_string())
else:
    print("No metrics found.")
