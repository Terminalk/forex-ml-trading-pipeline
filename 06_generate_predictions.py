import os
import logging
import pandas as pd
import numpy as np
import pickle
import json
from xgboost import XGBClassifier


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler('logs/generate_predictions.log', mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    _logger.addHandler(file_handler)
    _logger.addHandler(stream_handler)

    return _logger


logger = setup_logging()


def load_config(config_path="config_files/model_config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data_and_preprocess(file_path, feature_columns, window_size=5):
    logger.info(f"Loading: {file_path}")
    df = pd.read_parquet(file_path)
    logger.info(f"Rows loaded: {len(df)}")

    X = df[feature_columns].values

    def create_time_series(X, window_size):
        n_samples = len(X) - window_size
        n_features = X.shape[1]
        X_reshaped = np.zeros((n_samples, window_size * n_features))
        for i in range(n_samples):
            X_reshaped[i] = X[i:i + window_size].flatten()
        return X_reshaped

    X = create_time_series(X, window_size)
    df_reduced = df.iloc[window_size:].reset_index(drop=False).copy()

    logger.info(f"X shape after windowing: {X.shape}")
    return X, df_reduced


def main():
    logger.info("=" * 60)
    logger.info(" XGBOOST PREDICTION ".center(60, "="))
    logger.info("=" * 60)

    config = load_config("config_files/model_config.json")

    processed_data_dir = config["data"]["processed_data_dir"]
    test_file = os.path.join(processed_data_dir, config["data"]["test_file"])
    model_path = os.path.join("outputs", config["model"]["output_name"])
    feature_columns = config["model"]["features"]
    window_size = config["model"]["window_size"]

    logger.info(f"Model path:   {model_path}")
    logger.info(f"Test file:    {test_file}")
    logger.info(f"Window size:  {window_size}")
    logger.info(f"Features ({len(feature_columns)}): {feature_columns}")
    logger.info("-" * 60)

    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        return

    logger.info(f"Loading model: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")

    X_test, df_test = load_data_and_preprocess(test_file, feature_columns, window_size)

    logger.info("Running predictions...")
    y_pred_classes = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    df_test['predicted_signal'] = y_pred_classes
    for class_idx in range(y_pred_proba.shape[1]):
        df_test[f'predicted_probability_{class_idx}'] = y_pred_proba[:, class_idx]

    counts = pd.Series(y_pred_classes).value_counts().sort_index()
    for cls, cnt in counts.items():
        logger.info(f"  Class {cls}: {cnt} predictions ({cnt / len(y_pred_classes) * 100:.2f}%)")

    new_file_path = test_file.replace('.parquet', '_with_predictions.parquet')
    df_test.to_parquet(new_file_path)
    logger.info(f"Predictions saved to: {new_file_path}")

    logger.info("=" * 60)
    logger.info(" PREDICTION COMPLETED ".center(60, "="))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()