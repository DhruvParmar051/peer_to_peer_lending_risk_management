import os

import numpy as np
import pandas as pd

def merge_datasets():
    # Load datasets
    loan_data = pd.read_csv(os.path.join("data", "Loan_status_2007-2020Q3.gzip"), low_memory=False)
    unemployment_data = pd.read_csv(os.path.join("data", "unemployment_rate_by_state.csv"), low_memory=False)

    # Merge datasets on quarter and state
    merged_data = pd.merge(
        loan_data,
        unemployment_data,
        how='left',
        left_on=['addr_state', 'issue_d'],
        right_on=['addr_state', 'quarter']
    )

    # Save merged dataset
<<<<<<< HEAD
<<<<<<< HEAD
    output_dir = os.path.join("data", 'raw_data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'data.parquet')
    merged_data.to_parquet(output_path, index=False)
=======
    merged_data.to_csv(os.path.join("data", 'raw_data' ,"data.gzip"), index=False)
>>>>>>> 81b372d (data cleaning created)
=======
    output_dir = os.path.join("data", 'raw_data')
    os.makedirs(output_dir, exist_ok=True)
<<<<<<< HEAD
    output_path = os.path.join(output_dir, 'data.gzip')
    merged_data.to_csv(output_path, index=False)
>>>>>>> 54fff53 (update in data_merge)
=======
    output_path = os.path.join(output_dir, 'data.parquet')
    merged_data.to_parquet(output_path, index=False)
>>>>>>> 91a139b (pipeline updated)
    print("Saved data.csv")
    
merge_datasets()