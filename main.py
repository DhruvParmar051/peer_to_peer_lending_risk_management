
import os
import logging
import warnings

# Import your pipeline functions
from src.data_cleaning import clean_data_pipeline
from src.data_feature_engineering import feature_engineering_pipeline
from src.data_preprocessing import data_preprocessing_pipeline
from src.model import model_pipeline
from utils.logger import get_logger

warnings.filterwarnings("ignore")

logger = get_logger(__name__)

def main():
    logger.info("===== STARTING FULL ML PIPELINE =====")

    BASE_DIR = os.getcwd()

    # PATH CONFIGURATION
    RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw_data", "data.parquet")

    CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned_data")
    CLEANED_OUTPUT = os.path.join(CLEANED_DIR, "cleaned_data.parquet")

    FEATURE_DIR = os.path.join(BASE_DIR, "data", "feature_engineered")
    FEATURE_OUTPUT = os.path.join(FEATURE_DIR, "engineered_data.parquet")

    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

    MODEL_DIR = os.path.join(BASE_DIR, "models")

    # Ensure directories exist
    os.makedirs(CLEANED_DIR, exist_ok=True)
    os.makedirs(FEATURE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


    # STEP 1: cleaning
    logger.info("STEP 1: Running Data Cleaning Pipeline")
    clean_data_pipeline(RAW_DATA_PATH, CLEANED_DIR)

    # STEP 2: feature engineering
    logger.info("STEP 2: Running Feature Engineering Pipeline")
    feature_engineering_pipeline(CLEANED_OUTPUT, FEATURE_DIR)

    # STEP 3: preprocessing
    logger.info("STEP 3: Running Data Preprocessing Pipeline")
    data_preprocessing_pipeline(FEATURE_OUTPUT, PROCESSED_DIR)

    # STEP 4: Model Training + Tuning
    logger.info("STEP 4: Running Model Tuning Pipeline")
    model_pipeline(PROCESSED_DIR, MODEL_DIR)

    logger.info("===== FULL ML PIPELINE COMPLETED SUCCESSFULLY =====")


if __name__ == "__main__":
    main()
