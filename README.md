# 📈 Forex ML Trading System

> An end-to-end machine learning pipeline for forex signal prediction. Classifies each candle as **BUY / SELL / NONE** using an XGBoost model trained on MetaTrader 5 data.

---

## 🗂️ Project Structure

```
├── config_files/
│   ├── data_config.json            # Data download and labeling settings
│   ├── model_config.json           # Model and feature configuration
│   ├── backtest_config.json        # Backtest parameters
│   └── optuna_config.json          # Optuna optimization settings
│
├── 01_data_download_and_label.py   # Download data from MT5, generate targets
├── 02_feature_engineering.py       # Feature engineering (TA, price, time, lags)
├── 03_feature_scaling.py           # StandardScaler normalization
├── 04_generate_feature_list.py     # Export feature list to JSON
├── 05_train_model.py               # XGBoost training with early stopping
├── 06_generate_predictions.py      # Generate predictions on the test set
├── 07_backtest.py                  # Backtest engine with position management
│
└── optuna_optimization.py          # Optuna optimization (feature selection and/or hyperparameter tuning)
```

---

## ⚙️ Pipeline

### 1. Data Download & Labeling
Connects to MetaTrader 5 and fetches OHLCV data across three splits (train / validation / test). Each candle is assigned a forward-looking target:

- **1 (BUY)** — price hits TP before SL for a long position
- **2 (SELL)** — price hits TP before SL for a short position
- **0 (NONE)** — neither condition is met within the configured candle window

TP/SL parameters are set in `data_config.json`.

### 2. Feature Engineering
Features are generated across several groups:

| Group | Examples |
|---|---|
| TA-Lib indicators | SMA, EMA, RSI, ATR, MACD, ADX, MFI, WILLR |
| Price & candlestick | Body, Shadow, Doji, Hammer, Engulfing, Key Reversal |
| Volume analysis | VWAP, Volume Imbalance, Amihud Illiquidity, Relative Volume |
| Pivot Points & Fibonacci | PP, R1/S1/R2/S2, Fib levels 23.6%–100% |
| Market sessions | Asian / London / NY Session, London Open, NY Open |
| Time features | hour_sin/cos, month_sin/cos, weekday |
| Advanced | Price Autocorr, Spread Z-Score, Mean Reversion, Momentum Quality |
| Lags | Lags 1, 2, 3 for all numeric features |

### 3. Feature Scaling
`StandardScaler` is fitted exclusively on the training set, then applied to validation and test. The fitted scaler is saved as a `.pkl` file.

### 4. Feature Selection & Optimization
Optimization is powered by a single script `optuna_optimization.py`, controlled via `config_files/optuna_config.json`. The mode is selected by toggling the flags in the config:

- **Feature selection only** — `"optimize_features": true, "optimize_hyperparameters": false`
- **Hyperparameter tuning only** — `"optimize_features": false, "optimize_hyperparameters": true`
- **Joint optimization** — `"optimize_features": true, "optimize_hyperparameters": true`

Key config parameters:

```json
{
  "n_trials": 100,
  "min_trades": 80,
  "optimize_features": true,
  "optimize_hyperparameters": false,
  "hyperparameter_search_space": {
    "n_estimators":  { "type": "int",   "low": 100,   "high": 600  },
    "max_depth":     { "type": "int",   "low": 3,     "high": 8    },
    "learning_rate": { "type": "float", "low": 0.005, "high": 0.05, "log": true }
  }
}
```

### 5. Model Training
`XGBClassifier` with:
- Sliding time window (candles flattened into a single feature vector)
- Balanced class weights (`compute_class_weight("balanced")`)
- Early stopping on the validation set
- `multi:softmax` objective (3 classes)

### 6. Predictions
The model outputs `predicted_signal` (class 0/1/2) and per-class probabilities, appended directly to the `.parquet` file.

### 7. Backtesting
Backtest engine featuring:
- Trading hours filter (e.g. 9:00–17:00)
- Day-of-week filter
- Maximum concurrent open positions limit
- Configurable TP/SL separate from training labels
- Optional trend filters (EMA position, SMA cross, candle ratio)
- Multi-range reporting (monthly, quarterly, custom periods)

---

## 🚀 Quick Start

### Requirements

```bash
pip install pandas numpy xgboost scikit-learn optuna joblib ta-lib MetaTrader5
```

### Configuration

Edit the files in `config_files/`:

```json
{
  "symbol": "EURUSD",
  "timeframe": "mt5.TIMEFRAME_H1",
  "final_tp_pips": 50,
  "sl_pips": 25,
  "train": { "start": {"year": 2020, "month": 1, "day": 1} }
}
```

### Run

```bash
# 1. Download data and generate targets
python 01_data_download_and_label.py

# 2. Feature engineering
python 02_feature_engineering.py

# 3. Scale features
python 03_feature_scaling.py

# 4. (Optional) Optimize features and/or hyperparameters
python optuna_optimization.py

# 5. Train the model
python 05_train_model.py

# 6. Generate predictions
python 06_generate_predictions.py

# 7. Run backtest
python 07_backtest.py
```

---

## 📁 Directory Layout

| Directory | Contents |
|---|---|
| `original_data/` | Raw data from MT5 |
| `processed_data/` | Data after feature engineering and scaling |
| `outputs/` | Trained model `.pkl` and scaler |
| `results/` | Backtest results |
| `features_lists/` | Selected feature list `.json` |
| `logs/` | Logs for each pipeline stage |

---

## 📝 Notes

- Script `01_*` requires a running and logged-in MetaTrader 5 client.
- All intermediate files are stored in **Parquet** format (Apache Arrow).
- This project is intended for **research and educational purposes** and does not constitute investment advice.