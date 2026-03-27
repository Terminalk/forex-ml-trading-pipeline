# -*- coding: utf-8 -*-

import os
import logging
import pandas as pd
import json


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler('logs/generate_features_list.log', mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    _logger.addHandler(file_handler)
    _logger.addHandler(stream_handler)

    return _logger


logger = setup_logging()

input_file = 'processed_data/train_with_target_processed.parquet'
output_folder = 'features_lists'
output_file = os.path.join(output_folder, 'features_list.json')

exclude_columns = {
    'open', 'close', 'high', 'low', 'spread', 'time',
    'tick_volume', 'real_volume', 'test_buy', 'test_sell', 'target'
}

logger.info("=" * 60)
logger.info(" GENERATING FEATURES LIST ".center(60, "="))
logger.info("=" * 60)

logger.info(f"Loading: {input_file}")
df = pd.read_parquet(input_file)

all_columns = df.columns.tolist()
column_names = [
    col for col in all_columns
    if col not in exclude_columns and 'target' not in col.lower()
]

logger.info(f"Total columns found:   {len(all_columns)}")
logger.info(f"Excluded columns:      {len(all_columns) - len(column_names)}")
logger.info(f"Feature columns saved: {len(column_names)}")

os.makedirs(output_folder, exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(column_names, f, ensure_ascii=False, indent=2)

logger.info(f"Saved to: {output_file}")
logger.info(f"Columns: {column_names}")
logger.info("=" * 60)