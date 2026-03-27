import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import logging


def load_config(config_file):
    with open(config_file, "r") as f:
        return json.load(f)


def setup_logging(log_folder, log_file_name):
    os.makedirs(log_folder, exist_ok=True)
    log_path = os.path.join(log_folder, log_file_name)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    logging.info(f"Logging started. Log file: {log_path}")


def get_data_between_dates(symbol, timeframe, date_from, date_to):
    rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
    if rates is None:
        logging.warning(f"No data for {symbol} in range {date_from} - {date_to}")
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    logging.info(f"Fetched {len(df)} records for {symbol}")
    return df


def save_data(df, folder, filename):
    if df.empty:
        logging.warning(f"No data to save: {filename}")
        return
    os.makedirs(folder, exist_ok=True)
    output_path = os.path.join(folder, filename)
    df.reset_index().to_parquet(output_path, engine="pyarrow", index=False)
    logging.info(f"Data saved to: {output_path} ({len(df)} records)")


def collect_data_from_mt5(config):
    symbol = config["symbol"]
    timeframe = eval(config["timeframe"])
    output_folder = config.get("output_folder", "original_data")

    if not mt5.initialize():
        logging.error("Failed to connect to MetaTrader 5.")
        return False

    try:
        train_conf = config["train"]
        train_start = datetime(**train_conf["start"])
        train_end = datetime(**train_conf["end"], hour=23, minute=59, second=59)
        logging.info(f"Fetching TRAIN data: {train_start} - {train_end}")
        train_data = get_data_between_dates(symbol, timeframe, train_start, train_end)
        save_data(train_data, output_folder, train_conf["filename"])

        validation_conf = config["validation"]
        validation_start = datetime(**validation_conf["start"])
        validation_end = datetime(**validation_conf["end"], hour=23, minute=59, second=59)
        logging.info(f"Fetching VALIDATION data: {validation_start} - {validation_end}")
        validation_data = get_data_between_dates(symbol, timeframe, validation_start, validation_end)
        save_data(validation_data, output_folder, validation_conf["filename"])

        test_conf = config["test"]
        test_start = datetime(**test_conf["start"])
        test_end = datetime(**test_conf["end"], hour=23, minute=59, second=59)
        logging.info(f"Fetching TEST data: {test_start} - {test_end}")
        test_data = get_data_between_dates(symbol, timeframe, test_start, test_end)
        save_data(test_data, output_folder, test_conf["filename"])

        return True

    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
        return False
    except Exception as e:
        logging.exception(f"An error occurred: {e}")
        return False
    finally:
        mt5.shutdown()
        logging.info("MetaTrader 5 connection closed.")


def configure_trading(final_tp_pips, sl_pips, pip_value, default_spread_pips,
                      max_candles_duration, use_max_candles_duration):
    return {
        'final_tp_pips': final_tp_pips,
        'sl_pips': sl_pips,
        'pip_value': pip_value,
        'default_spread_pips': default_spread_pips,
        'max_candles_duration': max_candles_duration,
        'use_max_candles_duration': use_max_candles_duration,
    }


def calculate_pips(entry_price, exit_price, direction, pip_value):
    if pip_value == 0:
        return 0
    if direction == 'buy':
        return (exit_price - entry_price) / pip_value
    elif direction == 'sell':
        return (entry_price - exit_price) / pip_value
    else:
        return 0


def prepare_data(input_file_path):
    try:
        df = pd.read_parquet(input_file_path)
        logging.info(f"Data loaded. Row count: {len(df)}")
    except FileNotFoundError:
        logging.error(f"File not found: {input_file_path}")
        return None
    except Exception as e:
        logging.error(f"Error reading Parquet file: {e}")
        return None

    if 'time' not in df.columns:
        logging.error("Missing 'time' column in data.")
        return None

    try:
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            if pd.api.types.is_numeric_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'], unit='ns', errors='coerce')
            else:
                df['time'] = pd.to_datetime(df['time'], errors='coerce')

        if df['time'].isnull().any():
            logging.error("'time' column contains invalid datetime values (NaT) after conversion.")
            return None

        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
    except Exception as e:
        logging.error(f"Cannot process 'time' column as datetime index: {e}")
        return None

    required_columns = ['open', 'high', 'low', 'close', 'spread']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing required columns: {', '.join(missing_cols)}")
        return None

    return df


def calculate_targets(df, config):
    use_max_candles = config['use_max_candles_duration']
    max_candles = config['max_candles_duration']

    if use_max_candles:
        logging.info(f"Calculating target column (max_candles_duration={max_candles})...")
    else:
        logging.info("Calculating target column (max_candles_duration disabled — scanning to end of data)...")

    pip_value = config['pip_value']
    tp_pips = config['final_tp_pips']
    sl_pips = config['sl_pips']
    default_spread = config['default_spread_pips']

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    spreads = df['spread'].values
    n = len(df)

    targets = np.zeros(n, dtype=int)

    for i in range(n):
        spread_pips = spreads[i]
        if pd.isna(spread_pips) or not isinstance(spread_pips, (int, float)) or spread_pips < 0:
            spread_pips = default_spread

        entry_price = closes[i]

        if use_max_candles:
            end_idx = min(i + 1 + max_candles, n)
        else:
            end_idx = n

        buy_tp = entry_price + tp_pips * pip_value
        buy_sl = entry_price - sl_pips * pip_value

        buy_hit_tp = False
        buy_hit_sl = False
        for j in range(i + 1, end_idx):
            if lows[j] <= buy_sl:
                buy_hit_sl = True
                break
            if highs[j] >= buy_tp:
                buy_hit_tp = True
                break

        sell_tp = entry_price - tp_pips * pip_value
        sell_sl = entry_price + sl_pips * pip_value

        sell_hit_tp = False
        sell_hit_sl = False
        for j in range(i + 1, end_idx):
            if highs[j] >= sell_sl:
                sell_hit_sl = True
                break
            if lows[j] <= sell_tp:
                sell_hit_tp = True
                break

        if buy_hit_tp and not sell_hit_tp:
            targets[i] = 1
        elif sell_hit_tp and not buy_hit_tp:
            targets[i] = 2
        elif buy_hit_tp and sell_hit_tp:
            targets[i] = 0

    df['target'] = targets

    buy_count = int((targets == 1).sum())
    sell_count = int((targets == 2).sum())
    none_count = int((targets == 0).sum())
    logging.info(f"Target distribution: BUY={buy_count}, SELL={sell_count}, NONE={none_count}")

    return df


def process_file(file_path, file_type, config, input_data_folder):
    logging.info("\n" + "=" * 60)
    logging.info(f" Processing {file_type.upper()} ".center(60, "="))
    logging.info("=" * 60)

    df = prepare_data(file_path)
    if df is None:
        logging.error(f"Failed to load data for {file_type}.")
        return False

    df = calculate_targets(df, config)

    output_file = os.path.join(input_data_folder, f'{file_type}_with_target.parquet')
    df.reset_index().to_parquet(output_file, engine='pyarrow', index=False)

    buy_count = len(df[df['target'] == 1])
    sell_count = len(df[df['target'] == 2])
    total = buy_count + sell_count

    logging.info(f"File saved: {output_file}")
    logging.info(f"Total signals: {total}  |  BUY (target=1): {buy_count}  |  SELL (target=2): {sell_count}")
    logging.info("=" * 60)

    return True


if __name__ == "__main__":
    CONFIG_FILE = "config_files/data_config.json"

    try:
        config = load_config(CONFIG_FILE)
    except FileNotFoundError:
        print(f"[ERROR] Configuration file '{CONFIG_FILE}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parsing error: {e}")
        exit(1)

    LOG_FOLDER = config.get("log_folder", "logs")
    LOG_FILE = config.get("log_file", "data_download_and_label.log")
    setup_logging(LOG_FOLDER, LOG_FILE)

    logging.info(f"Configuration loaded from: {CONFIG_FILE}")

    COLLECT_DATA = config.get("collect_data", True)
    INPUT_DATA_FOLDER = config.get("input_data_folder", "original_data")
    TRAIN_FILE = config.get("train_file", "train.parquet")
    VALIDATION_FILE = config.get("validation_file", "validation.parquet")
    TEST_FILE = config.get("test_file", "test.parquet")

    try:
        trading_config = configure_trading(
            final_tp_pips=config.get("final_tp_pips", 40),
            sl_pips=config.get("sl_pips", 30),
            pip_value=config.get("pip_value", 0.0001),
            default_spread_pips=config.get("default_spread_pips", 1.0),
            max_candles_duration=config.get("max_candles_duration", 4000),
            use_max_candles_duration=config.get("use_max_candles_duration", True),
        )
        logging.info("Trading configuration initialized successfully.")
        logging.info(f"Final TP: {trading_config['final_tp_pips']} pips  |  SL: {trading_config['sl_pips']} pips")
        if trading_config['use_max_candles_duration']:
            logging.info(f"Max candles duration: {trading_config['max_candles_duration']} (ENABLED)")
        else:
            logging.info("Max candles duration: DISABLED (scanning to end of data)")
    except ValueError as config_error:
        logging.error(f"Trading configuration error: {config_error}")
        exit(1)

    if COLLECT_DATA:
        logging.info("=" * 60)
        logging.info(" FETCHING DATA FROM MT5 ".center(60, "="))
        logging.info("=" * 60)
        if not collect_data_from_mt5(config):
            logging.error("Failed to fetch data from MT5. Exiting.")
            exit(1)

    logging.info("\n" + "=" * 60)
    logging.info(" CREATING TARGET COLUMN ".center(60, "="))
    logging.info("=" * 60)

    if not os.path.isdir(INPUT_DATA_FOLDER):
        os.makedirs(INPUT_DATA_FOLDER, exist_ok=True)
        logging.info(f"Created data folder: '{INPUT_DATA_FOLDER}'")

    train_file_path = os.path.join(INPUT_DATA_FOLDER, TRAIN_FILE)
    validation_file_path = os.path.join(INPUT_DATA_FOLDER, VALIDATION_FILE)
    test_file_path = os.path.join(INPUT_DATA_FOLDER, TEST_FILE)

    for path, label in [(train_file_path, 'train'), (validation_file_path, 'validation'), (test_file_path, 'test')]:
        if not os.path.isfile(path):
            logging.critical(f"File '{path}' not found.")
            exit(1)

    logging.info(f"Data folder:       {INPUT_DATA_FOLDER}")
    logging.info(f"Train file:        {TRAIN_FILE}")
    logging.info(f"Validation file:   {VALIDATION_FILE}")
    logging.info(f"Test file:         {TEST_FILE}")
    logging.info("-" * 60)

    results = []
    for path, label in [
        (train_file_path, 'train'),
        (validation_file_path, 'validation'),
        (test_file_path, 'test')
    ]:
        ok = process_file(path, label, trading_config, INPUT_DATA_FOLDER)
        results.append((label, ok))

    logging.info("\n" + "=" * 60)
    logging.info(" FINAL SUMMARY ".center(60, "="))
    logging.info("=" * 60)
    for label, ok in results:
        status = "OK" if ok else "ERROR"
        logging.info(f"  {label.upper():<12} {status}")
    logging.info("=" * 60)