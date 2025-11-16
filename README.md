# 🚀 Peer-to-Peer Lending Risk Management

A complete, modular, end-to-end machine learning system for predicting **loan default risk** in peer-to-peer (P2P) lending platforms.  
The project includes **data cleaning**, **feature engineering**, **leakage-free preprocessing**, **model training**, and **evaluation** using Lending Club–style datasets (2007–2020).

---

## 📌 Table of Contents

- 📄 Overview  
- ⭐ Features  
- 📁 Project Structure  
- ⚙️ Installation  
- ▶️ Usage  
- 🧠 Pipeline Details  
- 📊 Outputs  
- 🔮 Future Improvements  
- 📜 License  

---

## 📄 Overview

This project builds a fully reproducible credit-risk modeling workflow.  
It provides a complete pipeline with:

- 🧹 **Robust data cleaning** & missing value handling  
- 📉 **Hybrid IQR outlier capping** with summary reports  
- 🧠 **Feature engineering** for loan metadata, borrower traits, ratios, and time features  
- 🧪 **Leakage-free ML preprocessing** (scaling, encoding inside pipelines)  
- 🤖 **XGBoost model** with hyperparameter tuning  
- 📦 **Train/Test parquet datasets**, **trained models**, and **logs**  
- 📊 **EDA notebooks** for insights  

Designed for: **fintech research**, **ML coursework**, and **risk analytics demos**.

---

## ⭐ Features

- 🗂 **Cleaned / engineered / processed datasets** stored as Parquet  
- 🧮 **Hybrid IQR capping** to handle extreme values  
- 🔁 **PCA and non-PCA engineered features**  
- 🔒 **Leakage-free ML pipeline**  
- 🧠 **XGBoost tuned model** with stored parameters  
- 📝 **Detailed pipeline logging**  
- 📓 **Jupyter notebooks for exploration**  

---

## 📁 Project Structure

```plaintext
peer_to_peer_lending_risk_management/
├── requirements.txt
├── main.py
├── .gitignore
├── LICENSE
├── Report/
|   ├── Project_Report_Peer_to_Peer_Lending_Risk_Management.pdf   
├── data/
│   ├── Loan_status_2007-2020Q3.gzip
│   ├── unemployment_rate_by_state.csv
│   │
│   ├── cleaned_data/
│   │   ├── cleaned_data.parquet
│   │   ├── hybrid_capping_summary.csv
│   │   └── hybrid_capping_summary_final.csv
│   │
│   ├── feature_engineered/
│   │   ├── engineered_data.parquet
│   │   └── engineered_data_no_pca.parquet
│   │
│   ├── processed/
│   │   ├── X_train_processed.parquet
│   │   ├── X_test_processed.parquet
│   │   ├── y_train.parquet
│   │   └── y_test.parquet
│   │
│   └── raw_data/
│       └── data.parquet
│
├── logs/
│   └── pipeline.log
│
├── models/
│   ├── xgboost_tuned.pkl
│   └── xgboost_tuning_results.csv
│
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   └── work.ipynb
│
├── src/
│   ├── data_cleaning.py
│   ├── data_feature_engineering.py
│   ├── data_preprocessing.py
│   ├── model.py
│   └── __init__.py
│
└── utils/
    ├── data_fetch.py
    ├── data_load.py
    ├── data_merge.py
    ├── hybrid_iqr_capping.py
    ├── logger.py
    └── __init__.py
```

---

## ⚙️ Installation

1. **Clone this repo** and enter the directory:
    ```
    git clone https://github.com/DhruvParmar051/peer_to_peer_lending_risk_management.git
    cd peer_to_peer_lending_risk_management
    ```

2. **(Recommended) Create a virtual environment**:
    ```
    python -m venv venv
    source venv/bin/activate       # On Windows: venv\Scripts\activate
    ```

3. **Install requirements**:
    ```
    pip install -r requirements.txt
    ```

---

## ▶️ Running the Full Pipeline

The project includes a **`main.py` pipeline orchestrator** that automatically runs all machine-learning pipeline components in sequence:

- 🧹 **Data Cleaning**  
- 🏗 **Feature Engineering**  
- ⚙️ **Preprocessing**  
- 🤖 **Model Training (with tuning)**  

Running the entire ML workflow requires only:

```
python main.py
```
This executes the full credit-risk modeling pipeline from raw data to trained model.

---

## 🔍 What `main.py` Does Internally

The orchestrator sequentially calls the following pipelines:

- clean_data_pipeline(...)  
- feature_engineering_pipeline(...)  
- data_preprocessing_pipeline(...)  
- model_pipeline(...)  

Each step automatically:
- Logs progress
- Saves intermediate datasets  
- Writes artifacts to the appropriate directories  

---

## ▶️ Running Individual Pipeline Steps (Optional)

If you prefer to run components separately:

### 1️⃣ Data Cleaning
```
from src.data_cleaning import clean_data_pipeline  
clean_data_pipeline(input_path, output_dir)
```


### 2️⃣ Feature Engineering
```
from src.data_feature_engineering import feature_engineering_pipeline  
feature_engineering_pipeline(cleaned_file, output_dir)
```

### 3️⃣ Preprocessing
```
from src.data_preprocessing import data_preprocessing_pipeline  
data_preprocessing_pipeline(feature_file, output_dir)
```

### 4️⃣ Model Training
```
from src.model import model_pipeline  
model_pipeline(processed_dir, model_dir)
```

---

## 🧠 Pipeline Details

### 🧹 Data Cleaning
- Missing value handling  
- Hybrid IQR outlier detection & capping  
- Summary report generation  
- Outputs: cleaned_data.parquet  

### 🏗 Feature Engineering
- Numerical & categorical transformations  
- PCA / non-PCA feature variants  
- Outputs: engineered_data.parquet  

### ⚙️ Preprocessing
- Scalers and encoders inside Scikit-learn pipelines  
- Strictly leakage-free transformations  
- Outputs: train-test parquet files  

### 🤖 Model Training
- XGBoost model with HalvingGridSearchCV tuning  
- Saves final trained model as .pkl  
- Saves tuning results as .csv  

---

## 📊 Outputs

| Folder | Description |
|--------|-------------|
| data/cleaned_data/ | Cleaned dataset + IQR capping summary |
| data/feature_engineered/ | PCA engineered datasets |
| data/processed/ | Train/test processed datasets |
| models/ | Final model + tuning results |
| logs/ | Full pipeline logs |

---

## 🔮 Future Improvements

- 🧾 Add SHAP explainability  
- 🖥 Build Streamlit dashboard for risk scoring  
- 🌐 Add FastAPI model-serving endpoint  
- 🔁 Add model drift monitoring + alerts  
- 🧬 Add LightGBM, CatBoost, and ensemble models  
- ⚙️ Move pipeline to Airflow / Prefect  

---

## 📜 License

MIT License
