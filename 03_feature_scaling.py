import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.preprocessing import StandardScaler


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler('logs/feature_scaling.log', mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    _logger.addHandler(file_handler)
    _logger.addHandler(stream_handler)

    return _logger


logger = setup_logging()

train_file = "processed_data/train_with_target_processed.parquet"
validation_file = "processed_data/validation_with_target_processed.parquet"
test_file = "processed_data/test_with_target_processed.parquet"

train_scaled_file = "processed_data/train_scaled.parquet"
validation_scaled_file = "processed_data/validation_scaled.parquet"
test_scaled_file = "processed_data/test_scaled.parquet"

scaler_file = "outputs/scaler_features.pkl"

columns_to_exclude = [
    "open", "close", "high", "low", "spread", "time",
    "tick_volume", "real_volume", "test_buy", "test_sell", "target"
]

logger.info("=" * 60)
logger.info(" TRAIN ".center(60, "="))
logger.info("=" * 60)

logger.info(f"Loading: {train_file}")
train_df = pd.read_parquet(train_file)
logger.info(f"Rows loaded: {len(train_df)}")

train_features = train_df.drop(columns=columns_to_exclude, errors="ignore")

inf_columns = train_features.columns[np.isinf(train_features).any()].tolist()
nan_columns = train_features.columns[train_features.isna().any()].tolist()

if inf_columns:
    logger.warning(f"Columns containing infinities (train): {inf_columns}")
if nan_columns:
    logger.warning(f"Columns containing NaN (train): {nan_columns}")

train_features.replace([np.inf, -np.inf], np.nan, inplace=True)
train_features.fillna(train_features.mean(), inplace=True)

train_non_scaled = train_df.loc[:, train_df.columns.intersection(columns_to_exclude)]

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_features)
scaler.feature_names = train_features.columns.tolist()

train_scaled_df = pd.DataFrame(train_scaled, columns=train_features.columns, index=train_df.index)
train_final_df = pd.concat([train_scaled_df, train_non_scaled], axis=1)

train_final_df.to_parquet(train_scaled_file, index=False)

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
joblib.dump(scaler, scaler_file)

logger.info(f"Scaler fitted on {len(scaler.feature_names)} features")
logger.info(f"Scaler saved: {scaler_file}")
logger.info(f"Train scaled file saved: {train_scaled_file}")

logger.info("=" * 60)
logger.info(" VALIDATION ".center(60, "="))
logger.info("=" * 60)

logger.info(f"Loading: {validation_file}")
validation_df = pd.read_parquet(validation_file)
logger.info(f"Rows loaded: {len(validation_df)}")

validation_features = validation_df.drop(columns=columns_to_exclude, errors="ignore")
validation_non_scaled = validation_df.loc[:, validation_df.columns.intersection(columns_to_exclude)]

scaler = joblib.load(scaler_file)

inf_columns = validation_features.columns[np.isinf(validation_features).any()].tolist()
nan_columns = validation_features.columns[validation_features.isna().any()].tolist()

if inf_columns:
    logger.warning(f"Columns containing infinities (validation): {inf_columns}")
if nan_columns:
    logger.warning(f"Columns containing NaN (validation): {nan_columns}")

validation_features.replace([np.inf, -np.inf], np.nan, inplace=True)
validation_features.fillna(validation_features.mean(), inplace=True)

validation_scaled = scaler.transform(validation_features)

validation_scaled_df = pd.DataFrame(validation_scaled, columns=scaler.feature_names, index=validation_df.index)
validation_final_df = pd.concat([validation_scaled_df, validation_non_scaled], axis=1)

validation_final_df.to_parquet(validation_scaled_file, index=False)
logger.info(f"Validation scaled file saved: {validation_scaled_file}")

logger.info("=" * 60)
logger.info(" TEST ".center(60, "="))
logger.info("=" * 60)

logger.info(f"Loading: {test_file}")
test_df = pd.read_parquet(test_file)
logger.info(f"Rows loaded: {len(test_df)}")

test_features = test_df.drop(columns=columns_to_exclude, errors="ignore")
test_non_scaled = test_df.loc[:, test_df.columns.intersection(columns_to_exclude)]

scaler = joblib.load(scaler_file)

inf_columns = test_features.columns[np.isinf(test_features).any()].tolist()
nan_columns = test_features.columns[test_features.isna().any()].tolist()

if inf_columns:
    logger.warning(f"Columns containing infinities (test): {inf_columns}")
if nan_columns:
    logger.warning(f"Columns containing NaN (test): {nan_columns}")

test_features.replace([np.inf, -np.inf], np.nan, inplace=True)
test_features.fillna(test_features.mean(), inplace=True)

test_scaled = scaler.transform(test_features)

test_scaled_df = pd.DataFrame(test_scaled, columns=scaler.feature_names, index=test_df.index)
test_final_df = pd.concat([test_scaled_df, test_non_scaled], axis=1)

test_final_df.to_parquet(test_scaled_file, index=False)
logger.info(f"Test scaled file saved: {test_scaled_file}")

logger.info("=" * 60)
logger.info(" SUMMARY ".center(60, "="))
logger.info("=" * 60)
logger.info(f"  {'TRAIN':<12} {train_scaled_file}")
logger.info(f"  {'VALIDATION':<12} {validation_scaled_file}")
logger.info(f"  {'TEST':<12} {test_scaled_file}")
logger.info(f"  {'SCALER':<12} {scaler_file}")
logger.info("Normalization completed. All transformed files have been saved.")
logger.info("=" * 60)