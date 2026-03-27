import os
import talib
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Degrees of freedom')


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler('logs/feature_engineering.log', mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    _logger.addHandler(file_handler)
    _logger.addHandler(stream_handler)

    return _logger


logger = setup_logging()


def add_ta_features(df):
    logger.info("Adding TA-Lib indicators...")
    new_cols = {}
    new_cols['SMA_5'] = talib.SMA(df['close'], timeperiod=5)
    new_cols['SMA_10'] = talib.SMA(df['close'], timeperiod=10)
    new_cols['EMA_20'] = talib.EMA(df['close'], timeperiod=20)
    new_cols['EMA_50'] = talib.EMA(df['close'], timeperiod=50)
    new_cols['EMA_200'] = talib.EMA(df['close'], timeperiod=200)
    new_cols['RSI'] = talib.RSI(df['close'], timeperiod=14)
    new_cols['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(df['close'])
    new_cols['MACD'] = macd
    new_cols['MACD_signal'] = macd_signal
    new_cols['MACD_hist'] = macd_hist
    new_cols['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    new_cols['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['tick_volume'], timeperiod=14)
    new_cols['WILLR'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_time_features(df):
    logger.info("Adding time features...")
    new_cols = {}
    new_cols['hour'] = df['time'].dt.hour
    new_cols['minute'] = df['time'].dt.minute
    new_cols['weekday'] = df['time'].dt.weekday
    new_cols['month'] = df['time'].dt.month
    new_cols['day'] = df['time'].dt.day
    new_cols['is_weekend'] = df['time'].dt.weekday.isin([5, 6]).astype(int)

    new_cols['hour_sin'] = np.sin(2 * np.pi * new_cols['hour'] / 24)
    new_cols['hour_cos'] = np.cos(2 * np.pi * new_cols['hour'] / 24)
    new_cols['month_sin'] = np.sin(2 * np.pi * new_cols['month'] / 12)
    new_cols['month_cos'] = np.cos(2 * np.pi * new_cols['month'] / 12)

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def calculate_pivot_points(df):
    logger.info("Adding Pivot Points...")
    new_cols = {}
    new_cols['PP'] = (df['high'] + df['low'] + df['close']) / 3
    new_cols['R1'] = 2 * new_cols['PP'] - df['low']
    new_cols['S1'] = 2 * new_cols['PP'] - df['high']
    new_cols['R2'] = new_cols['PP'] + (df['high'] - df['low'])
    new_cols['S2'] = new_cols['PP'] - (df['high'] - df['low'])

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def calculate_fibonacci_levels(df):
    logger.info("Adding Fibonacci Levels...")
    new_cols = {}
    new_cols['fib_23.6'] = df['close'] + 0.236 * (df['high'] - df['low'])
    new_cols['fib_38.2'] = df['close'] + 0.382 * (df['high'] - df['low'])
    new_cols['fib_50'] = df['close'] + 0.5 * (df['high'] - df['low'])
    new_cols['fib_61.8'] = df['close'] + 0.618 * (df['high'] - df['low'])
    new_cols['fib_100'] = df['close'] + 1 * (df['high'] - df['low'])

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_custom_features(df):
    logger.info("Adding custom features...")
    new_cols = {}
    df_temp = df.set_index('time')

    new_cols['Candle_Body'] = df['close'] - df['open']
    new_cols['Shadow'] = df['high'] - df['low']
    new_cols['Shadow_Body_Ratio'] = new_cols['Shadow'] / (abs(new_cols['Candle_Body']).replace(0, 1e-10))
    new_cols['Body_Range_Ratio'] = abs(new_cols['Candle_Body']) / (new_cols['Shadow'].replace(0, 1e-10))
    new_cols['Upper_Shadow'] = df['high'] - df[['close', 'open']].max(axis=1).values
    new_cols['Lower_Shadow'] = df[['close', 'open']].min(axis=1).values - df['low']
    new_cols['Shadow_Ratio'] = new_cols['Upper_Shadow'] / (new_cols['Lower_Shadow'].replace(0, 1e-10))

    new_cols['Volatility'] = df['close'].rolling(14, min_periods=1).std()
    new_cols['Volatility_MA'] = new_cols['Volatility'].rolling(14, min_periods=1).mean()
    new_cols['Price_ROC'] = df['close'].pct_change(7)
    new_cols['RSI_ROC'] = df['RSI'].diff(3)
    new_cols['ATR_Ratio'] = df['ATR'] / df['close']
    new_cols['Norm_Range'] = (df['high'] - df['low']) / df['close']
    new_cols['Volatility_Spike'] = new_cols['Volatility'] / new_cols['Volatility'].rolling(50, min_periods=1).mean()
    new_cols['EMA_cross'] = (df['EMA_20'] > df['EMA_50']).astype(int)
    new_cols['Trend_Confirmed'] = ((df['EMA_20'] > df['EMA_50']) & (df['close'] > df['EMA_50'])).astype(int)

    window = 50
    rolling_price_vol = (df['close'] * df['tick_volume']).rolling(window, min_periods=1).sum()
    rolling_vol = df['tick_volume'].rolling(window, min_periods=1).sum()
    new_cols['VWAP'] = rolling_price_vol / rolling_vol.replace(0, 1e-10)
    new_cols['VWAP_Distance'] = (df['close'] - new_cols['VWAP']) / df['close']
    new_cols['Volume_Upper_Half'] = ((df['close'] > (df['high'] + df['low']) / 2) * df['tick_volume'])
    new_cols['Volume_Lower_Half'] = ((df['close'] <= (df['high'] + df['low']) / 2) * df['tick_volume'])
    new_cols['Volume_Imbalance'] = (new_cols['Volume_Upper_Half'] - new_cols['Volume_Lower_Half']) / df[
        'tick_volume'].replace(0, 1e-10)
    new_cols['Relative_Volume'] = df['tick_volume'] / df['tick_volume'].rolling(20, min_periods=1).mean()
    new_cols['Effective_Spread'] = (df['high'] - df['low']) / df['close']
    new_cols['Spread_MA'] = new_cols['Effective_Spread'].rolling(20, min_periods=1).mean()
    new_cols['Spread_Volatility'] = new_cols['Effective_Spread'].rolling(20, min_periods=1).std()
    new_cols['Price_Impact'] = abs(df['close'].diff()) / df['tick_volume'].replace(0, 1e-10)
    new_cols['Liquidity_Proxy'] = df['tick_volume'] / new_cols['Effective_Spread'].replace(0, 1e-10)

    hl_range = (df['high'] - df['low']).replace(0, 1e-10)
    new_cols['Buy_Pressure'] = (df['close'] - df['low']) / hl_range
    new_cols['Sell_Pressure'] = (df['high'] - df['close']) / hl_range
    new_cols['Net_Pressure'] = new_cols['Buy_Pressure'] - new_cols['Sell_Pressure']

    new_cols['Volume_MA_Ratio'] = df['tick_volume'] / df['tick_volume'].rolling(20, min_periods=1).mean()
    new_cols['Price_Volume_Trend'] = new_cols['Candle_Body'] * df['tick_volume']

    new_cols['Is_Doji'] = (abs(new_cols['Candle_Body']) / new_cols['Shadow'].replace(0, 1e-10) < 0.1).astype(int)
    new_cols['Is_Hammer'] = ((new_cols['Lower_Shadow'] > 2 * abs(new_cols['Candle_Body'])) &
                             (new_cols['Upper_Shadow'] < 0.1 * new_cols['Shadow'])).astype(int)
    new_cols['Is_Shooting_Star'] = ((new_cols['Upper_Shadow'] > 2 * abs(new_cols['Candle_Body'])) &
                                    (new_cols['Lower_Shadow'] < 0.1 * new_cols['Shadow'])).astype(int)
    new_cols['Bullish_Engulfing'] = ((new_cols['Candle_Body'] > 0) &
                                     (new_cols['Candle_Body'].shift(1) < 0) &
                                     (abs(new_cols['Candle_Body']) > abs(new_cols['Candle_Body'].shift(1)))).astype(int)

    new_cols['Asian_Session'] = df['hour'].between(0, 8).astype(int)
    new_cols['London_Session'] = df['hour'].between(8, 16).astype(int)
    new_cols['NY_Session'] = df['hour'].between(13, 21).astype(int)
    new_cols['Overlap_Session'] = df['hour'].between(13, 16).astype(int)
    new_cols['London_Open'] = (df['hour'] == 8).astype(int)
    new_cols['NY_Open'] = (df['hour'] == 14).astype(int)
    new_cols['London_Close'] = (df['hour'] == 16).astype(int)
    new_cols['First_Candle_Hour'] = (df['minute'] == 0).astype(int)
    new_cols['Last_Candle_Hour'] = (df['minute'] == 55).astype(int)

    new_cols['Dist_From_SMA5'] = (df['close'] - df['SMA_5']) / df['close']
    new_cols['Dist_From_EMA20'] = (df['close'] - df['EMA_20']) / df['close']
    new_cols['Dist_From_PP'] = (df['close'] - df['PP']) / df['close']

    new_cols['Price_Zscore'] = (df['close'] - df['close'].rolling(50, min_periods=1).mean()) / df['close'].rolling(50,
                                                                                                                   min_periods=1).std().replace(
        0, 1e-10)
    new_cols['RSI_Zscore'] = (df['RSI'] - df['RSI'].rolling(50, min_periods=1).mean()) / df['RSI'].rolling(50,
                                                                                                           min_periods=1).std().replace(
        0, 1e-10)

    for period in [10, 20, 50, 100]:
        ma = df['close'].rolling(period, min_periods=1).mean()
        std = df['close'].rolling(period, min_periods=1).std().replace(0, 1e-10)
        new_cols[f'MeanReversion_{period}'] = (df['close'] - ma) / std
        new_cols[f'MR_Extreme_{period}'] = (abs(new_cols[f'MeanReversion_{period}']) > 2).astype(int)

    for period in [3, 5, 10]:
        new_cols[f'ROC_{period}'] = df['close'].pct_change(period)
        new_cols[f'RSI_ROC_{period}'] = df['RSI'].diff(period)

    new_cols['Price_Acceleration'] = df['close'].diff().diff()
    new_cols['Volume_Acceleration'] = df['tick_volume'].diff().diff()

    for period in [5, 10, 20]:
        roc = df['close'].pct_change(period)
        new_cols[f'ROC_Acceleration_{period}'] = roc.diff()
        new_cols[f'ROC_Deceleration_{period}'] = (roc.diff() < 0).astype(int)

    new_cols['Momentum_Strength'] = abs(df['close'].pct_change(10)) * df['ADX']
    new_cols['Price_Velocity'] = df['close'].diff() / df['close'].shift(1).replace(0, 1e-10)
    new_cols['Price_Jerk'] = new_cols['Price_Acceleration'].diff()

    bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'])
    new_cols['BB_upper'] = bb_upper
    new_cols['BB_middle'] = bb_middle
    new_cols['BB_lower'] = bb_lower
    new_cols['BB_Width'] = (bb_upper - bb_lower) / bb_middle.replace(0, 1e-10)
    bb_range = (bb_upper - bb_lower).replace(0, 1e-10)
    new_cols['BB_Position'] = (df['close'] - bb_lower) / bb_range

    for window in [20, 50, 100]:
        new_cols[f'Resistance_{window}'] = df['high'].rolling(window, min_periods=1).max()
        new_cols[f'Support_{window}'] = df['low'].rolling(window, min_periods=1).min()
        new_cols[f'Distance_To_Resistance_{window}'] = (new_cols[f'Resistance_{window}'] - df['close']) / df['close']
        new_cols[f'Distance_To_Support_{window}'] = (df['close'] - new_cols[f'Support_{window}']) / df['close']

    new_cols['Highest_20'] = df['high'].rolling(20, min_periods=1).max()
    new_cols['Lowest_20'] = df['low'].rolling(20, min_periods=1).min()
    range_20 = (new_cols['Highest_20'] - new_cols['Lowest_20']).replace(0, 1e-10)
    new_cols['Position_in_Range'] = (df['close'] - new_cols['Lowest_20']) / range_20

    new_cols['Breakout_High_20'] = (df['high'] > df['high'].rolling(20, min_periods=1).max().shift(1)).astype(int)
    new_cols['Breakout_Low_20'] = (df['low'] < df['low'].rolling(20, min_periods=1).min().shift(1)).astype(int)
    new_cols['Touching_Resistance'] = (abs(df['high'] - new_cols['Resistance_20']) / df['close'] < 0.0005).astype(int)
    new_cols['Touching_Support'] = (abs(df['low'] - new_cols['Support_20']) / df['close'] < 0.0005).astype(int)

    new_cols['Tick_Direction'] = np.sign(df['close'].diff())
    new_cols['Tick_Direction_Sum'] = new_cols['Tick_Direction'].rolling(10, min_periods=1).sum()

    delta = np.where(df['close'] > df['open'], df['tick_volume'], -df['tick_volume'])
    new_cols['Delta'] = pd.Series(delta, index=df.index)
    new_cols['Cumulative_Delta'] = new_cols['Delta'].rolling(20, min_periods=1).sum()
    new_cols['Delta_Divergence'] = (np.sign(df['close'].diff()) != np.sign(new_cols['Delta'].diff())).astype(int)

    vol_q75 = df['tick_volume'].rolling(10, min_periods=1).quantile(0.75)
    new_cols['Volume_Cluster_High'] = ((df['close'] > df['close'].shift(1)) & (df['tick_volume'] > vol_q75)).astype(int)
    new_cols['Volume_Cluster_Low'] = ((df['close'] < df['close'].shift(1)) & (df['tick_volume'] > vol_q75)).astype(int)

    new_cols['Absorption'] = ((new_cols['Effective_Spread'] < new_cols['Spread_MA']) &
                              (df['tick_volume'] > df['tick_volume'].rolling(10, min_periods=1).mean())).astype(int)

    vol_mean_5 = df['tick_volume'].rolling(5, min_periods=1).mean()
    new_cols['Aggressive_Buy'] = ((df['close'] >= df['open']) & (df['tick_volume'] > vol_mean_5)).astype(int)
    new_cols['Aggressive_Sell'] = ((df['close'] < df['open']) & (df['tick_volume'] > vol_mean_5)).astype(int)

    log_hl = np.log((df['high'] / df['low']).replace(0, 1e-10))
    new_cols['Parkinson_Vol'] = np.sqrt((1 / (4 * np.log(2))) * log_hl ** 2)
    new_cols['Parkinson_Vol_MA'] = new_cols['Parkinson_Vol'].rolling(20, min_periods=1).mean()

    log_co = np.log((df['close'] / df['open']).replace(0, 1e-10))
    new_cols['GK_Vol'] = np.sqrt(0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2)

    vol_quantiles = df['close'].rolling(100, min_periods=1).std().rolling(20, min_periods=1).quantile(0.75)
    new_cols['High_Vol_Regime'] = (df['close'].rolling(20, min_periods=1).std() > vol_quantiles).astype(int)
    new_cols['Vol_Clustering'] = new_cols['Volatility'].rolling(5, min_periods=1).std()

    hl_diff = (df['high'] - df['low']).replace(0, 1e-10)
    chaikin_ema = talib.EMA(hl_diff, timeperiod=10)
    new_cols['Chaikin_Vol'] = (chaikin_ema - chaikin_ema.shift(10)) / chaikin_ema.shift(10).replace(0, 1e-10) * 100

    vol_mean_20 = df['tick_volume'].rolling(20, min_periods=1).mean()
    new_cols['News_Spike'] = ((new_cols['Volatility'] > 2 * new_cols['Volatility_MA']) &
                              (df['tick_volume'] > 2 * vol_mean_20)).astype(int)

    new_cols['RSI_ATR_Ratio'] = df['RSI'] / (df['ATR'] * 100).replace(0, 1e-10)
    new_cols['Volume_Volatility'] = df['tick_volume'] * new_cols['Volatility']
    new_cols['MACD_RSI_Divergence'] = (np.sign(df['MACD'].diff()) != np.sign(df['RSI'].diff())).astype(int)
    new_cols['Trend_Strength'] = df['ADX'] * abs(df['MACD'])

    df_temp_calc = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    bullish_signals = (
            (df['RSI'] < 30).astype(int) +
            (df['close'] < df_temp_calc['BB_lower']).astype(int) +
            (df['MACD_hist'] > df['MACD_hist'].shift(1)).astype(int) +
            df_temp_calc['Is_Hammer'].astype(int) +
            (df['close'] < df_temp_calc['Support_20']).astype(int)
    )
    new_cols['Bullish_Confluence'] = bullish_signals

    bearish_signals = (
            (df['RSI'] > 70).astype(int) +
            (df['close'] > df_temp_calc['BB_upper']).astype(int) +
            (df['MACD_hist'] < df['MACD_hist'].shift(1)).astype(int) +
            df_temp_calc['Is_Shooting_Star'].astype(int) +
            (df['close'] > df_temp_calc['Resistance_20']).astype(int)
    )
    new_cols['Bearish_Confluence'] = bearish_signals

    new_cols['Market_Regime'] = np.where(
        (df['ADX'] > 25) & (df['EMA_20'] > df['EMA_50']), 2,
        np.where((df['ADX'] > 25) & (df['EMA_20'] < df['EMA_50']), 0, 1)
    )

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_advanced_features(df):
    logger.info("Adding advanced features...")
    new_cols = {}
    returns = df['close'].pct_change()

    for short, long in [(5, 20), (10, 50), (20, 100)]:
        mom_short = df['close'].pct_change(short)
        mom_long = df['close'].pct_change(long)
        new_cols[f'Momentum_Divergence_{short}_{long}'] = mom_short - mom_long
        new_cols[f'Momentum_Ratio_{short}_{long}'] = mom_short / (mom_long.abs().replace(0, 1e-10))

    new_cols['Consecutive_Up'] = (df['close'] > df['open']).astype(int).rolling(10, min_periods=1).sum()
    new_cols['Consecutive_Down'] = (df['close'] < df['open']).astype(int).rolling(10, min_periods=1).sum()
    new_cols['Up_Down_Ratio_10'] = new_cols['Consecutive_Up'] / (new_cols['Consecutive_Down'].replace(0, 1e-10))

    direction = np.sign(df['close'] - df['open'])
    streak = (direction != direction.shift(1)).astype(int).cumsum()
    new_cols['Current_Streak_Length'] = direction.groupby(streak).cumcount() + 1
    new_cols['Streak_Direction'] = direction

    for window in [10, 20, 50]:
        sq_returns = returns ** 2
        new_cols[f'Volatility_Cluster_{window}'] = sq_returns.rolling(window, min_periods=1).mean()

    for window in [20, 50]:
        downside_returns = returns.copy()
        downside_returns[downside_returns > 0] = np.nan
        upside_returns = returns.copy()
        upside_returns[upside_returns < 0] = np.nan

        downside_vol = downside_returns.rolling(window, min_periods=1).std()
        upside_vol = upside_returns.rolling(window, min_periods=1).std()
        new_cols[f'Vol_Asymmetry_{window}'] = downside_vol / upside_vol.replace(0, 1e-10)

    midpoint = (df['high'] + df['low']) / 2
    new_cols['Price_Above_Midpoint'] = (df['close'] > midpoint).astype(int)
    new_cols['Distance_From_Midpoint'] = (df['close'] - midpoint) / df['close']

    new_cols['Roll_Measure'] = -2 * returns.rolling(20, min_periods=2).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )

    for window in [20, 50]:
        abs_returns = returns.abs()
        new_cols[f'Amihud_Illiquidity_{window}'] = (abs_returns / df['tick_volume'].replace(0, 1e-10)).rolling(window,
                                                                                                               min_periods=1).mean()

    for window in [50, 100]:
        buy_vol = pd.Series(np.where(df['close'] > df['open'], df['tick_volume'], 0), index=df.index)
        sell_vol = pd.Series(np.where(df['close'] <= df['open'], df['tick_volume'], 0), index=df.index)
        total_vol = df['tick_volume'].rolling(window, min_periods=1).sum()
        vol_imbalance = (buy_vol - sell_vol).abs().rolling(window, min_periods=1).sum()
        new_cols[f'VPIN_Proxy_{window}'] = vol_imbalance / total_vol.replace(0, 1e-10)

    for window in [20, 50, 100]:
        rv_cc = (returns ** 2).rolling(window, min_periods=1).sum()
        new_cols[f'RV_Close_{window}'] = rv_cc

        hl_ratio = np.log(df['high'] / df['low'].replace(0, 1e-10))
        hl_var = (hl_ratio ** 2) / (4 * np.log(2))
        new_cols[f'RV_Parkinson_{window}'] = hl_var.rolling(window, min_periods=1).sum()

        log_hc = np.log(df['high'] / df['close'].replace(0, 1e-10))
        log_ho = np.log(df['high'] / df['open'].replace(0, 1e-10))
        log_lc = np.log(df['low'] / df['close'].replace(0, 1e-10))
        log_lo = np.log(df['low'] / df['open'].replace(0, 1e-10))
        rs_var = log_hc * log_ho + log_lc * log_lo
        new_cols[f'RV_RS_{window}'] = rs_var.rolling(window, min_periods=1).sum()

    for window in [20, 50]:
        abs_returns = returns.abs()
        bv = (np.pi / 2) * (abs_returns * abs_returns.shift(1)).rolling(window, min_periods=1).sum()
        rv = (returns ** 2).rolling(window, min_periods=1).sum()

        jump_component = (rv - bv).clip(lower=0)
        new_cols[f'Jump_Component_{window}'] = jump_component
        new_cols[f'Jump_Ratio_{window}'] = jump_component / rv.replace(0, 1e-10)

    rolling_std = returns.rolling(50, min_periods=1).std()
    new_cols['Price_Jump_3std'] = (returns.abs() > 3 * rolling_std).astype(int)
    new_cols['Price_Jump_5std'] = (returns.abs() > 5 * rolling_std).astype(int)

    for window in [20, 50]:
        buy_intensity = pd.Series(np.where(df['close'] > df['open'], df['tick_volume'], 0), index=df.index)
        sell_intensity = pd.Series(np.where(df['close'] <= df['open'], df['tick_volume'], 0), index=df.index)

        new_cols[f'Order_Toxicity_{window}'] = (
                (buy_intensity - sell_intensity).abs().rolling(window, min_periods=1).sum() /
                df['tick_volume'].rolling(window, min_periods=1).sum().replace(0, 1e-10)
        )

    for window in [20, 50, 100]:
        price_change = abs(df['close'] - df['close'].shift(window))
        path_length = abs(df['close'].diff()).rolling(window, min_periods=1).sum()
        new_cols[f'Efficiency_Ratio_{window}'] = price_change / path_length.replace(0, 1e-10)

    for window in [20, 50]:
        vol_weighted_ret = (returns * df['tick_volume']).rolling(window, min_periods=1).sum()
        total_vol = df['tick_volume'].rolling(window, min_periods=1).sum()
        new_cols[f'Vol_Weighted_Return_{window}'] = vol_weighted_ret / total_vol.replace(0, 1e-10)

    for window in [50, 100]:
        high_roll = df['high'].rolling(window, min_periods=1).max()
        low_roll = df['low'].rolling(window, min_periods=1).min()
        price_range = (high_roll - low_roll).replace(0, 1e-10)

        upper_third = low_roll + 2 * price_range / 3
        vol_upper = pd.Series(np.where(df['close'] > upper_third, df['tick_volume'], 0), index=df.index)
        new_cols[f'Vol_Upper_Third_{window}'] = vol_upper.rolling(window, min_periods=1).sum() / df[
            'tick_volume'].rolling(window, min_periods=1).sum()

        lower_third = low_roll + price_range / 3
        vol_lower = pd.Series(np.where(df['close'] < lower_third, df['tick_volume'], 0), index=df.index)
        new_cols[f'Vol_Lower_Third_{window}'] = vol_lower.rolling(window, min_periods=1).sum() / df[
            'tick_volume'].rolling(window, min_periods=1).sum()

    for window in [50, 100]:
        mean_ret = returns.rolling(window, min_periods=1).mean()
        downside_dev = pd.Series(np.where(returns < mean_ret, (returns - mean_ret) ** 2, 0), index=df.index)
        new_cols[f'Downside_Deviation_{window}'] = np.sqrt(downside_dev.rolling(window, min_periods=1).mean())

        new_cols[f'Return_Downside_Ratio_{window}'] = mean_ret / new_cols[f'Downside_Deviation_{window}'].replace(0,
                                                                                                                  1e-10)

    for window in [50, 100]:
        new_cols[f'VaR_1pct_{window}'] = returns.rolling(window, min_periods=1).quantile(0.01)
        new_cols[f'VaR_5pct_{window}'] = returns.rolling(window, min_periods=1).quantile(0.05)

    for window in [20, 50, 100]:
        new_cols[f'Price_Vol_Corr_{window}'] = df['close'].rolling(window, min_periods=2).corr(df['tick_volume'])
        new_cols[f'Return_Vol_Corr_{window}'] = returns.rolling(window, min_periods=2).corr(df['tick_volume'])

    for window in [20, 50]:
        new_cols[f'HighLow_Corr_{window}'] = df['high'].rolling(window, min_periods=2).corr(df['low'])

    for period in [10, 20, 50]:
        raw_momentum = df['close'].pct_change(period)
        smooth_momentum = df['close'].rolling(period, min_periods=1).mean().pct_change(period)
        new_cols[f'Momentum_Quality_{period}'] = smooth_momentum / raw_momentum.abs().replace(0, 1e-10)

    for period in [3, 5, 10]:
        new_cols[f'Reversal_Indicator_{period}'] = -returns.rolling(period, min_periods=1).sum()

    for window in [10, 20]:
        price_mom = df['close'].pct_change(window)
        vol_mom = df['tick_volume'].pct_change(window)
        new_cols[f'Contrarian_Signal_{window}'] = (np.sign(price_mom) != np.sign(vol_mom)).astype(int)

    for window in [50, 100]:
        rolling_var = returns.rolling(window, min_periods=1).var()
        var_threshold = rolling_var.rolling(window, min_periods=1).quantile(0.75)
        new_cols[f'High_Var_Regime_{window}'] = (rolling_var > var_threshold).astype(int)

    for window in [20, 50]:
        up_days = (returns > 0).rolling(window, min_periods=1).sum()
        new_cols[f'Trend_Consistency_{window}'] = abs(up_days - window / 2) / (window / 2)

    new_cols['Inside_Bar'] = ((df['high'] <= df['high'].shift(1)) &
                              (df['low'] >= df['low'].shift(1))).astype(int)

    new_cols['Outside_Bar'] = ((df['high'] > df['high'].shift(1)) &
                               (df['low'] < df['low'].shift(1))).astype(int)

    body_size = abs(df['close'] - df['open'])
    total_range = (df['high'] - df['low']).replace(0, 1e-10)
    wick_ratio = body_size / total_range
    new_cols['Pin_Bar'] = (wick_ratio < 0.3).astype(int)

    new_cols['Bull_Key_Reversal'] = ((df['close'] > df['high'].shift(1)) &
                                     (df['low'] < df['low'].shift(1))).astype(int)
    new_cols['Bear_Key_Reversal'] = ((df['close'] < df['low'].shift(1)) &
                                     (df['high'] > df['high'].shift(1))).astype(int)

    for window in [10, 20, 50]:
        vw_momentum = (df['close'].pct_change() * df['tick_volume']).rolling(window, min_periods=1).sum()
        total_volume = df['tick_volume'].rolling(window, min_periods=1).sum()
        new_cols[f'VW_Momentum_{window}'] = vw_momentum / total_volume.replace(0, 1e-10)

    for window in [10, 20, 50]:
        tr = df['high'] - df['low']
        tr_ma = tr.rolling(window, min_periods=1).mean()
        new_cols[f'Range_Expansion_{window}'] = tr / tr_ma.replace(0, 1e-10)

        range_position = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, 1e-10)
        new_cols[f'Range_Position_Std_{window}'] = range_position.rolling(window, min_periods=1).std()

    for lag in [1, 5, 10]:
        new_cols[f'Price_Autocorr_Lag{lag}'] = df['close'].rolling(50, min_periods=lag + 1).apply(
            lambda x: x.autocorr(lag=lag) if len(x) > lag else 0, raw=False
        )

    for lag in [1, 5]:
        new_cols[f'Volume_Autocorr_Lag{lag}'] = df['tick_volume'].rolling(50, min_periods=lag + 1).apply(
            lambda x: x.autocorr(lag=lag) if len(x) > lag else 0, raw=False
        )

    spread = df['high'] - df['low']
    for window in [10, 20, 50]:
        new_cols[f'Spread_MA_{window}'] = spread.rolling(window, min_periods=1).mean()
        new_cols[f'Spread_Std_{window}'] = spread.rolling(window, min_periods=1).std()
        new_cols[f'Spread_Zscore_{window}'] = (spread - new_cols[f'Spread_MA_{window}']) / new_cols[
            f'Spread_Std_{window}'].replace(0, 1e-10)

    for window in [10, 20, 50]:
        mom = df['close'].pct_change(window)
        mom_consistent = (mom * mom.shift(1) > 0).astype(int)
        new_cols[f'Momentum_Persistence_{window}'] = mom_consistent.rolling(window, min_periods=1).mean()

    logger.info(f"Added {len(new_cols)} unique advanced features")
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_all_lags(df, lags=[1, 2, 3]):
    logger.info("Adding lag variables...")
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if '_lag' not in c]
    lagged_dict = {}

    for c in numeric_cols:
        for lag in lags:
            lagged_dict[f"{c}_lag{lag}"] = df[c].shift(lag)

    lagged_df = pd.DataFrame(lagged_dict, index=df.index)
    df = pd.concat([df, lagged_df], axis=1)

    df = df.loc[:, ~df.columns.duplicated()]
    return df


def process_data(file_paths):
    for file_path in file_paths:
        df = pd.read_parquet(file_path)
        logger.info(f"Processing data from file {file_path}...")

        df['time'] = pd.to_datetime(df['time'])

        df = add_ta_features(df)
        df = add_time_features(df)
        df = calculate_pivot_points(df)
        df = calculate_fibonacci_levels(df)
        df = add_custom_features(df)
        df = add_advanced_features(df)
        df = add_all_lags(df, lags=[1, 2, 3])

        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            logger.warning(f"Found {len(nan_cols)} columns with NaN before filling")
            nan_counts = df[nan_cols].isna().sum()
            logger.info(f"Top 10 columns with most NaN values:\n{nan_counts.nlargest(10)}")

        logger.info("Filling missing values...")

        df = df.ffill()
        df = df.bfill()
        df = df.fillna(0)

        remaining_nans = df.isna().sum().sum()
        if remaining_nans > 0:
            logger.error(f"WARNING: {remaining_nans} NaN values remaining after filling!")
        else:
            logger.info("All NaN values have been filled")

        output_file_path = Path("processed_data") / f"{Path(file_path).stem}_processed.parquet"
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_file_path, index=False)
        logger.info(f"Data saved to {output_file_path}")


def main():
    logger.info("Starting process...")
    file_paths = [
        "original_data/train_with_target.parquet",
        "original_data/validation_with_target.parquet",
        "original_data/test_with_target.parquet"
    ]
    process_data(file_paths)
    logger.info("Process completed successfully")


if __name__ == "__main__":
    main()