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
    merged_data.to_csv(os.path.join("data", "data.gzip"), index=False)
    print("Saved data.csv")
    
merge_datasets()