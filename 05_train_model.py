import pandas as pd
import numpy as np
import pickle
import json
import os
import logging
import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight


def load_config(config_path="config_files/model_config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_logging(config):
    log_dir = config["logging"]["log_dir"]
    log_file = config["logging"]["log_file"]
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logger initialized. Logs are being saved to: {log_path}")
    return log_path


def preprocess_data(df, feature_columns, window_size):
    X = df[feature_columns].values

    def create_time_series(X, window_size):
        n_samples = len(X) - window_size
        n_features = X.shape[1]
        X_reshaped = np.zeros((n_samples, window_size * n_features))
        for i in range(n_samples):
            X_reshaped[i] = X[i:i + window_size].flatten()
        return X_reshaped

    X = create_time_series(X, window_size)
    y = df['target'].values[window_size:]

    return X, y


def train_and_save_model(df_train, df_val, config):
    window_size = config["model"]["window_size"]
    early_stopping_rounds = config["model"]["early_stopping_rounds"]
    model_name = config["model"]["output_name"]
    feature_columns = config["model"]["features"]

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    model_output_path = os.path.join(output_dir, model_name)

    model_params = {k: v for k, v in config["model"]["params"].items()}
    model_params['eval_metric'] = 'mlogloss'
    model_params['early_stopping_rounds'] = early_stopping_rounds

    X_train, y_train = preprocess_data(df_train, feature_columns, window_size)
    X_val, y_val = preprocess_data(df_val, feature_columns, window_size)

    logging.info(f"X_train shape: {X_train.shape}  |  X_val shape: {X_val.shape}")

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    logging.info(f"Class weights: {class_weight_dict}")
    sample_weights = np.array([class_weight_dict[label] for label in y_train])

    model = XGBClassifier(**model_params)
    logging.info("Training the model...")

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    with open(model_output_path, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Model saved to: {model_output_path}")

    logging.info("Evaluating the model on the validation set...")
    y_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred)
    logging.info(f"Validation accuracy: {val_accuracy:.4f}")
    logging.info("\nClassification Report:\n" + classification_report(y_val, y_pred))

    if hasattr(model, 'best_iteration') and model.best_iteration is not None:
        logging.info(f"Best iteration (early stopping): {model.best_iteration}")


def main():
    config = load_config("config_files/model_config.json")
    setup_logging(config)

    logging.info("=" * 60)
    logging.info(" XGBOOST TRAINING ".center(60, "="))
    logging.info("=" * 60)
    logging.info(f"XGBoost version: {xgboost.__version__}")

    processed_data_dir = config["data"]["processed_data_dir"]
    train_file = config["data"]["train_file"]
    val_file = config["data"]["validation_file"]

    train_file_path = os.path.join(processed_data_dir, train_file)
    val_file_path = os.path.join(processed_data_dir, val_file)

    if not os.path.exists(train_file_path):
        logging.error(f"Training file not found: {train_file_path}")
        return
    if not os.path.exists(val_file_path):
        logging.error(f"Validation file not found: {val_file_path}")
        return

    logging.info(f"Loading train:      {train_file_path}")
    df_train = pd.read_parquet(train_file_path)
    logging.info(f"Loading validation: {val_file_path}")
    df_val = pd.read_parquet(val_file_path)

    if 'time' in df_train.columns:
        df_train = df_train.sort_values("time").reset_index(drop=True)
    if 'time' in df_val.columns:
        df_val = df_val.sort_values("time").reset_index(drop=True)

    logging.info(f"Training set size:   {df_train.shape}")
    logging.info(f"Validation set size: {df_val.shape}")
    logging.info(f"Features used ({len(config['model']['features'])}): {config['model']['features']}")
    logging.info(f"Window size: {config['model']['window_size']}")
    logging.info("-" * 60)

    train_and_save_model(df_train, df_val, config)

    logging.info("=" * 60)
    logging.info(" TRAINING COMPLETED ".center(60, "="))
    logging.info("=" * 60)


if __name__ == "__main__":
    main()