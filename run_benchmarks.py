import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import os
import time

notebooks_to_run = [
    '02_logistic_regression',
    '03_xgboost',
    '04_lightgbm',
    '05_gru_model',
    '06_transformer_model',
    '07_uplift_causal_model',
    '08_model_comparison'
]

os.chdir('notebooks')

for nb_name in notebooks_to_run:
    print(f"Running {nb_name}.ipynb...")
    start_time = time.time()
    try:
        with open(f"{nb_name}.ipynb") as f:
            nb = nbf.read(f, as_version=4)
        
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': '.'}})
        
        with open(f"{nb_name}_executed.ipynb", 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        
        print(f"Finished {nb_name}.ipynb in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"Error running {nb_name}.ipynb: {e}")
