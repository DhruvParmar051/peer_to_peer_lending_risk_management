import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def verify_winsorization(df_before, df_after, sample_cols=3):
    """
    Verify that winsorization worked correctly:
    - Checks no rows were dropped
    - Compares summary statistics (1%, 50%, 99%)
    - Visualizes before/after for sample numeric columns
    """
    print("=== Winsorization Verification ===")
    print(f"Rows before: {df_before.shape[0]}, after: {df_after.shape[0]}")
    print(f"Columns before: {df_before.shape[1]}, after: {df_after.shape[1]}")
    
    if df_before.shape[0] != df_after.shape[0]:
        print("⚠️ Warning: Some rows were removed — this should not happen with winsorization!")
    else:
        print("✅ No rows removed during winsorization.")

    # Select numeric columns
    num_cols = df_after.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        print("❌ No numeric columns found for verification.")
        return
    
    # Pick sample columns to show
    sample_cols = min(sample_cols, len(num_cols))
    selected = np.random.choice(num_cols, sample_cols, replace=False)
    
    print(f"\n🔍 Checking columns: {list(selected)}")
    for col in selected:
        print(f"\n=== {col} ===")
        desc_before = df_before[col].describe([0.01, 0.5, 0.99])
        desc_after = df_after[col].describe([0.01, 0.5, 0.99])
        print("Before Winsorization:\n", desc_before)
        print("After Winsorization:\n", desc_after)
        print("-" * 40)

    # Visualization (boxplots)
    print("\n📊 Generating before/after boxplots for visual comparison...")
    fig, axes = plt.subplots(sample_cols, 2, figsize=(10, 4 * sample_cols))
    if sample_cols == 1:
        axes = np.array([axes])
    for i, col in enumerate(selected):
        sns.boxplot(x=df_before[col], ax=axes[i, 0], color="lightcoral")
        axes[i, 0].set_title(f"{col} — Before")
        sns.boxplot(x=df_after[col], ax=axes[i, 1], color="lightgreen")
        axes[i, 1].set_title(f"{col} — After")
    plt.tight_layout()
    plt.show()

    # Quantile sanity check
    for col in num_cols:
        q1 = df_after[col].quantile(0.01)
        q99 = df_after[col].quantile(0.99)
        if not df_after[col].between(q1, q99).all():
            print(f"⚠️ {col} has values outside the capped range.")
    print("\n✅ Winsorization verification complete.")
