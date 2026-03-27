import pandas as pd
import numpy as np
import json
from datetime import time, timedelta, datetime
import os
import logging
import sys

CONFIG_FILE_PATH = 'config_files/backtest_config.json'
OUTPUTS_DIR = 'outputs'


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    _logger = logging.getLogger("backtest")
    _logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler('logs/backtest.log', mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    _logger.addHandler(file_handler)
    _logger.addHandler(stream_handler)

    return _logger


logger = setup_logging()


class TrendCriteria:
    def __init__(self, use_price_change: bool, use_sma_cross: bool,
                 use_candle_ratio: bool, use_ema_position: bool):
        self.use_price_change = use_price_change
        self.use_sma_cross = use_sma_cross
        self.use_candle_ratio = use_candle_ratio
        self.use_ema_position = use_ema_position


def load_config(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        logger.info(f"Configuration successfully loaded from file: {file_path}")
    except FileNotFoundError:
        logger.error(f"Error: Configuration file '{file_path}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Error: Invalid JSON format in file: {file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error while loading JSON file: {e}")
        sys.exit(1)

    try:
        trade_start_time = time.fromisoformat(config_data['start_hour'])
        trade_end_time = time.fromisoformat(config_data['end_hour'])

        trading_config = {
            'trade_start_time': trade_start_time,
            'trade_end_time': trade_end_time,
            'final_tp_pips': float(config_data['tp']),
            'sl_pips': float(config_data['sl']),
            'pip_value': float(config_data['pip_value']),
            'default_spread_pips': float(config_data['default_spread']),
            'max_open_positions': int(config_data['max_open_positions']),
            'trade_days': config_data['trade_days'],
            'buy_allow': int(config_data['buy_allow']),
            'sell_allow': int(config_data['sell_allow']),
        }

        input_folder = config_data['input_folder']
        file_name = config_data['file_name']
        results_folder = config_data['results_folder']
        date_ranges_to_test = config_data['data_ranges']
        required_columns = config_data['required_columns']
        signal_column = config_data['signal_column']

        trend_c_data = config_data['trend_criteria']
        trend_criteria_config = TrendCriteria(
            use_price_change=trend_c_data['use_price_change'],
            use_sma_cross=trend_c_data['use_sma_cross'],
            use_candle_ratio=trend_c_data['use_candle_ratio'],
            use_ema_position=trend_c_data['use_ema_position'],
        )

        return (trading_config, trend_criteria_config, input_folder, file_name,
                results_folder, date_ranges_to_test, required_columns, signal_column)

    except KeyError as e:
        logger.error(f"CONFIG ERROR: Missing required key in JSON: {e}.")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"CONFIG ERROR: Invalid data type: {e}.")
        sys.exit(1)


def calculate_pips(entry_price, exit_price, direction, pip_value):
    if pip_value == 0:
        return 0
    if direction == 'buy':
        return (exit_price - entry_price) / pip_value
    elif direction == 'sell':
        return (entry_price - exit_price) / pip_value
    return 0


def determine_trend_v2(df, current_index, lookback_candles=50, criteria_config=None):
    if criteria_config is None:
        return 'neutral'
    try:
        current_position = df.index.get_loc(current_index)
        if current_position < lookback_candles:
            return 'neutral'

        historical_data = df.iloc[current_position - lookback_candles:current_position]
        if len(historical_data) < lookback_candles:
            return 'neutral'

        bullish_signals = 0
        bearish_signals = 0

        if criteria_config.use_price_change:
            first_close = historical_data.iloc[0]['close']
            last_close = historical_data.iloc[-1]['close']
            pct = ((last_close - first_close) / first_close) * 100
            if pct > 0.5:
                bullish_signals += 1
            elif pct < -0.5:
                bearish_signals += 1

        if criteria_config.use_sma_cross:
            closes = historical_data['close'].values
            sma_short = np.mean(closes[-10:])
            sma_long = np.mean(closes[-30:])
            if sma_short > sma_long * 1.001:
                bullish_signals += 1
            elif sma_short < sma_long * 0.999:
                bearish_signals += 1

        if criteria_config.use_candle_ratio:
            green_candles = sum(1 for _, c in historical_data.iterrows() if c['close'] > c['open'])
            green_ratio = green_candles / lookback_candles
            if green_ratio > 0.6:
                bullish_signals += 1
            elif green_ratio < 0.4:
                bearish_signals += 1

        if criteria_config.use_ema_position:
            last_close = historical_data.iloc[-1]['close']
            ema_200 = df.loc[current_index]['EMA_200']
            if last_close > ema_200:
                bullish_signals += 1
            elif last_close < ema_200:
                bearish_signals += 1

        active_criteria = sum([
            criteria_config.use_price_change,
            criteria_config.use_sma_cross,
            criteria_config.use_candle_ratio,
            criteria_config.use_ema_position,
        ])
        required_signals = 1 if active_criteria <= 2 else 2

        if bullish_signals >= required_signals and bearish_signals == 0:
            return 'bullish'
        elif bearish_signals >= required_signals and bullish_signals == 0:
            return 'bearish'
        return 'neutral'

    except Exception as e:
        logger.error(f"Error while determining trend: {e}")
        return 'neutral'


def is_signal_aligned_with_trend(signal, trend):
    if trend == 'neutral':
        return True
    if signal == 1 and trend == 'bullish':
        return True
    if signal == 2 and trend == 'bearish':
        return True
    return False


def log_config(config):
    logger.info(f"Trading hours:               {config['trade_start_time']} - {config['trade_end_time']}")
    logger.info(f"Final TP (pips):             {config['final_tp_pips']}")
    logger.info(f"Initial SL (pips):           {config['sl_pips']}")
    logger.info(f"Spread:                      From 'spread' column (fallback: {config['default_spread_pips']} pips)")
    logger.info(f"Pip Value:                   {config['pip_value']}")
    logger.info(f"Trading days:                {', '.join(map(str, config['trade_days']))}")
    logger.info(f"Sell Allowed:                {config['sell_allow']}")
    logger.info(f"Buy Allowed:                 {config['buy_allow']}")
    logger.info(f"Maximum open positions:      {config['max_open_positions']}")


def log_config_to_file(config, f):
    f.write(f"Trading hours:               {config['trade_start_time']} - {config['trade_end_time']}\n")
    f.write(f"Final TP (pips):             {config['final_tp_pips']}\n")
    f.write(f"Initial SL (pips):           {config['sl_pips']}\n")
    f.write(f"Spread:                      From 'spread' column (fallback: {config['default_spread_pips']} pips)\n")
    f.write(f"Pip Value:                   {config['pip_value']}\n")
    f.write(f"Trading days:                {', '.join(map(str, config['trade_days']))}\n")
    f.write(f"Sell Allowed:                {config['sell_allow']}\n")
    f.write(f"Buy Allowed:                 {config['buy_allow']}\n")
    f.write(f"Maximum open positions:      {config['max_open_positions']}\n")


def run_backtest(start_date, end_date, file_suffix, input_file_path, output_folder,
                 config, trend_config, required_columns, signal_column):
    logger.info(f"--- Starting backtest for: {file_suffix} ({start_date} - {end_date}) ---")
    logger.info(f"Data file: {input_file_path}")

    os.makedirs(output_folder, exist_ok=True)
    log_file_path = os.path.join(output_folder, f"log_{file_suffix}.txt")

    results_file = open(log_file_path, 'w', encoding='utf-8')

    def w(line=""):
        results_file.write(line + "\n")

    try:
        try:
            df = pd.read_parquet(input_file_path)
            logger.info(f"Data loaded. Number of rows: {len(df)}")
        except FileNotFoundError:
            logger.error(f"File {input_file_path} was not found.")
            return None
        except Exception as e:
            logger.error(f"Error loading Parquet file: {e}")
            return None

        if 'time' not in df.columns:
            logger.error("Missing 'time' column in the data.")
            return None

        try:
            if not pd.api.types.is_datetime64_any_dtype(df['time']):
                if pd.api.types.is_numeric_dtype(df['time']):
                    df['time'] = pd.to_datetime(df['time'], unit='ns', errors='coerce')
                else:
                    df['time'] = pd.to_datetime(df['time'], errors='coerce')
            if df['time'].isnull().any():
                logger.error("The 'time' column contains invalid datetime values (NaT).")
                return None
            df.set_index('time', inplace=True)
            df.sort_index(inplace=True)
        except Exception as e:
            logger.error(f"Cannot process 'time' column as datetime index: {e}")
            return None

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {', '.join(missing_cols)}")
            return None
        if signal_column not in df.columns:
            logger.error(f"Missing signal column '{signal_column}' in the data.")
            return None

        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            df_filtered = df[(df.index >= start_dt) & (df.index <= end_dt)]
        except Exception as e:
            logger.error(f"Error filtering dates: {e}")
            return None

        if df_filtered.empty:
            logger.info("No data in the given date range. Backtest was not executed.")
            return None

        logger.info(f"Data filtered. Number of rows: {len(df_filtered)}")

        open_trades = []
        closed_trades = []
        trade_id_counter = 0
        current_equity_pips = 0.0
        peak_equity_pips = 0.0
        max_drawdown_pips = 0.0
        max_concurrent_trades = 0
        invalid_spread_count = 0
        trading_days = set()
        days_with_trades = set()
        trend_filter_stats = {
            'total_signals': 0,
            'signals_passed': 0,
            'signals_filtered': 0,
            'bullish_trend_count': 0,
            'bearish_trend_count': 0,
            'neutral_trend_count': 0,
        }

        logger.info("Starting historical data processing...")

        for index, row in df_filtered.iterrows():
            current_time = index
            current_high = row['high']
            current_low = row['low']
            current_close = row['close']
            signal = row[signal_column]

            if ('spread' in row and pd.notna(row['spread'])
                    and isinstance(row['spread'], (int, float)) and row['spread'] >= 0):
                current_spread_pips = row['spread']
            else:
                current_spread_pips = config['default_spread_pips']
                if 'spread' not in required_columns:
                    invalid_spread_count += 1

            spread_value = current_spread_pips * config['pip_value']
            trades_to_remove = []
            temp_equity_change = 0.0

            for trade in list(open_trades):
                exit_price = None
                outcome = None
                close_trade = False

                if trade['direction'] == 'buy':
                    if current_low <= trade['sl_price']:
                        exit_price, outcome, close_trade = trade['sl_price'], 'SL', True
                    elif current_high >= trade['final_tp_price']:
                        exit_price, outcome, close_trade = trade['final_tp_price'], 'TP', True
                elif trade['direction'] == 'sell':
                    if current_high >= trade['sl_price']:
                        exit_price, outcome, close_trade = trade['sl_price'], 'SL', True
                    elif current_low <= trade['final_tp_price']:
                        exit_price, outcome, close_trade = trade['final_tp_price'], 'TP', True

                if close_trade:
                    pnl_pips = calculate_pips(trade['entry_price_no_spread'], exit_price,
                                              trade['direction'], config['pip_value'])
                    closed_trades.append({
                        'id': trade['id'],
                        'entry_time': trade['entry_time'],
                        'exit_time': current_time,
                        'entry_price': trade['entry_price'],
                        'entry_price_no_spread': trade['entry_price_no_spread'],
                        'exit_price': exit_price,
                        'direction': trade['direction'],
                        'pnl_pips': pnl_pips,
                        'outcome': outcome,
                        'duration': current_time - trade['entry_time'],
                        'trend_at_entry': trade.get('trend_at_entry', 'unknown'),
                    })
                    trades_to_remove.append(trade)
                    temp_equity_change += pnl_pips

            if temp_equity_change != 0.0:
                current_equity_pips += temp_equity_change
                peak_equity_pips = max(peak_equity_pips, current_equity_pips)
                max_drawdown_pips = max(max_drawdown_pips, peak_equity_pips - current_equity_pips)

            open_trades = [t for t in open_trades if t not in trades_to_remove]

            is_within_hours = config['trade_start_time'] <= current_time.time() <= config['trade_end_time']
            is_trading_day = current_time.weekday() in config['trade_days']

            if is_within_hours and is_trading_day:
                trading_days.add(current_time.date())

                if signal in [1, 2] and len(open_trades) < config['max_open_positions']:
                    trend_filter_stats['total_signals'] += 1
                    current_trend = determine_trend_v2(df, current_time, lookback_candles=50,
                                                       criteria_config=trend_config)

                    if current_trend == 'bullish':
                        trend_filter_stats['bullish_trend_count'] += 1
                    elif current_trend == 'bearish':
                        trend_filter_stats['bearish_trend_count'] += 1
                    else:
                        trend_filter_stats['neutral_trend_count'] += 1

                    if is_signal_aligned_with_trend(signal, current_trend):
                        trend_filter_stats['signals_passed'] += 1
                        entry_price_no_spread = current_close

                        if signal == 1 and config['buy_allow'] == 1:
                            trade_id_counter += 1
                            open_trades.append({
                                'id': trade_id_counter,
                                'entry_time': current_time,
                                'entry_price': entry_price_no_spread + spread_value,
                                'entry_price_no_spread': entry_price_no_spread,
                                'direction': 'buy',
                                'final_tp_price': entry_price_no_spread + config['final_tp_pips'] * config['pip_value'],
                                'sl_price': entry_price_no_spread - config['sl_pips'] * config['pip_value'],
                                'trend_at_entry': current_trend,
                            })
                            days_with_trades.add(current_time.date())

                        elif signal == 2 and config['sell_allow'] == 1:
                            trade_id_counter += 1
                            open_trades.append({
                                'id': trade_id_counter,
                                'entry_time': current_time,
                                'entry_price': entry_price_no_spread - spread_value,
                                'entry_price_no_spread': entry_price_no_spread,
                                'direction': 'sell',
                                'final_tp_price': entry_price_no_spread - config['final_tp_pips'] * config['pip_value'],
                                'sl_price': entry_price_no_spread + config['sl_pips'] * config['pip_value'],
                                'trend_at_entry': current_trend,
                            })
                            days_with_trades.add(current_time.date())
                    else:
                        trend_filter_stats['signals_filtered'] += 1

            max_concurrent_trades = max(max_concurrent_trades, len(open_trades))

        if open_trades and not df_filtered.empty:
            logger.info(f"Closing {len(open_trades)} remaining trades at end of period ({end_date})...")
            last_close_price = df_filtered.iloc[-1]['close']
            last_time = df_filtered.index[-1]
            temp_equity_change = 0.0
            for trade in open_trades:
                pnl_pips = calculate_pips(trade['entry_price_no_spread'], last_close_price,
                                          trade['direction'], config['pip_value'])
                closed_trades.append({
                    'id': trade['id'],
                    'entry_time': trade['entry_time'],
                    'exit_time': last_time,
                    'entry_price': trade['entry_price'],
                    'entry_price_no_spread': trade['entry_price_no_spread'],
                    'exit_price': last_close_price,
                    'direction': trade['direction'],
                    'pnl_pips': pnl_pips,
                    'outcome': 'EndOfTest',
                    'duration': last_time - trade['entry_time'],
                    'trend_at_entry': trade.get('trend_at_entry', 'unknown'),
                })
                temp_equity_change += pnl_pips
            if temp_equity_change != 0.0:
                current_equity_pips += temp_equity_change
                peak_equity_pips = max(peak_equity_pips, current_equity_pips)
                max_drawdown_pips = max(max_drawdown_pips, peak_equity_pips - current_equity_pips)
            open_trades = []

        logger.info("Historical data processing completed.")
        if invalid_spread_count > 0:
            logger.warning(f"Default spread used {invalid_spread_count} times due to invalid 'spread' column data.")

        trades_df = None
        if closed_trades:
            trades_df = pd.DataFrame(closed_trades)
            if 'duration' in trades_df.columns:
                trades_df['duration_readable'] = trades_df['duration'].apply(lambda x: str(x).split('.')[0])

            total_trades = len(trades_df)
            profitable_trades = trades_df[trades_df['pnl_pips'] > 0]
            unprofitable_trades = trades_df[trades_df['pnl_pips'] < 0]
            neutral_trades = trades_df[trades_df['pnl_pips'] == 0]
            total_pnl_pips = trades_df['pnl_pips'].sum()

            max_tp_streak = current_tp_streak = 0
            max_sl_streak = current_sl_streak = 0
            for outcome in trades_df['outcome']:
                if outcome == 'TP':
                    current_tp_streak += 1
                    max_tp_streak = max(max_tp_streak, current_tp_streak)
                    current_sl_streak = 0
                elif outcome == 'SL':
                    current_sl_streak += 1
                    max_sl_streak = max(max_sl_streak, current_sl_streak)
                    current_tp_streak = 0
                else:
                    current_tp_streak = current_sl_streak = 0

            lossy_days = 0
            if 'exit_time' in trades_df.columns and trades_df['exit_time'].notna().any():
                trades_df['exit_date'] = trades_df['exit_time'].dt.date
                daily_pnl = trades_df.groupby('exit_date')['pnl_pips'].sum()
                lossy_days = (daily_pnl < 0).sum()

            average_duration_td = trades_df['duration'].mean() if total_trades > 0 else timedelta(0)
            average_duration_str = str(average_duration_td).split('.')[0] if pd.notna(average_duration_td) else "N/A"

            total_profit_pips = profitable_trades['pnl_pips'].sum()
            total_loss_pips = abs(unprofitable_trades['pnl_pips'].sum())
            profit_factor = (total_profit_pips / total_loss_pips if total_loss_pips > 0
                             else (np.inf if total_profit_pips > 0 else 0))

            total_weeks = lossy_weeks = 0
            if 'exit_time' in trades_df.columns and trades_df['exit_time'].notna().any():
                try:
                    iso = trades_df['exit_time'].dt.isocalendar()
                    trades_df['exit_year_week'] = (iso['year'].astype(str) + '-'
                                                   + iso['week'].astype(str).str.zfill(2))
                    weekly_pnl = trades_df.groupby('exit_year_week')['pnl_pips'].sum()
                    total_weeks = weekly_pnl.count()
                    lossy_weeks = (weekly_pnl < 0).sum()
                except Exception:
                    pass
        else:
            total_trades = 0
            total_pnl_pips = 0.0
            profitable_trades = pd.DataFrame()
            unprofitable_trades = pd.DataFrame()
            neutral_trades = pd.DataFrame()
            max_tp_streak = max_sl_streak = 0
            lossy_days = 0
            average_duration_str = "N/A"
            profit_factor = 0
            total_weeks = lossy_weeks = 0

        days_without_trades = len(trading_days - days_with_trades)
        win_rate = len(profitable_trades) / total_trades if total_trades > 0 else 0

        summary_results = {
            'period': file_suffix,
            'start_date': start_date,
            'end_date': end_date,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl_pips': total_pnl_pips,
            'max_drawdown_pips': max_drawdown_pips,
        }

        logger.info(f"Saving results to: {log_file_path}...")
        try:
            w("=" * 75)
            w(f" BACKTEST RESULTS: {file_suffix} ".center(75, "="))
            w("=" * 75)
            w(f"Period:          {start_date}  to  {end_date}")
            w(f"Generated on:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            w(f"Data file:       {input_file_path}")
            w()

            w("-" * 30 + " Configuration " + "-" * 30)
            log_config_to_file(config, results_file)
            w("Active trend criteria:")
            w(f"  - Price change:        {trend_config.use_price_change}")
            w(f"  - SMA cross:           {trend_config.use_sma_cross}")
            w(f"  - Candle ratio:        {trend_config.use_candle_ratio}")
            w(f"  - Position vs EMA:     {trend_config.use_ema_position}")
            w()

            w("-" * 30 + " General Statistics " + "-" * 25)
            if total_trades == 0:
                w("No closed trades to analyze.")
            else:
                loss_rate = len(unprofitable_trades) / total_trades
                w(f"Total number of trades:         {total_trades}")
                w(f"Profitable trades:              {len(profitable_trades)} ({win_rate:.2%})")
                w(f"Unprofitable trades:            {len(unprofitable_trades)} ({loss_rate:.2%})")
                if len(neutral_trades) > 0:
                    w(f"Neutral trades (BE):            {len(neutral_trades)}")
                w(f"Total P/L:                      {total_pnl_pips:.2f} pips")
                if len(profitable_trades) > 0:
                    w(f"Average profit per trade:       {profitable_trades['pnl_pips'].mean():.2f} pips")
                if len(unprofitable_trades) > 0:
                    w(f"Average loss per trade:         {unprofitable_trades['pnl_pips'].mean():.2f} pips")
                w(f"Max TP streak:                  {max_tp_streak}")
                w(f"Max SL streak:                  {max_sl_streak}")
                w(f"Maximum drawdown:               {max_drawdown_pips:.2f} pips")
                w(f"Max concurrent open trades:     {max_concurrent_trades}")
                w(f"Number of losing days:          {lossy_days}")
                w(f"Average trade duration:         {average_duration_str}")
                w(f"Profit factor:                  {profit_factor:.2f}")
                w(f"Weeks in backtest with trades:  {total_weeks}")
                w(f"Losing weeks:                   {lossy_weeks}")
                w(f"Days without trades:            {days_without_trades}")

            w()
            w("-" * 30 + " Trend Filter Statistics " + "-" * 21)
            w(f"Total signals:                  {trend_filter_stats['total_signals']}")
            w(f"Signals aligned with trend:     {trend_filter_stats['signals_passed']}")
            w(f"Signals filtered out:           {trend_filter_stats['signals_filtered']}")
            w(f"Bullish trend signals:          {trend_filter_stats['bullish_trend_count']}")
            w(f"Bearish trend signals:          {trend_filter_stats['bearish_trend_count']}")
            w(f"Neutral trend signals:          {trend_filter_stats['neutral_trend_count']}")
            w()

            w("-" * 30 + " Trade Details " + "-" * 30)
            w()
            if trades_df is not None and not trades_df.empty:
                cols_to_save = ['id', 'entry_time', 'exit_time', 'direction',
                                'entry_price_no_spread', 'entry_price', 'exit_price',
                                'pnl_pips', 'outcome', 'duration_readable']
                existing_cols = [col for col in cols_to_save if col in trades_df.columns]

                formatters = {}
                if 'pnl_pips' in existing_cols:
                    formatters['pnl_pips'] = '{:.2f}'.format
                if 'entry_price_no_spread' in existing_cols:
                    formatters['entry_price_no_spread'] = '{:.5f}'.format
                if 'entry_price' in existing_cols:
                    formatters['entry_price'] = '{:.5f}'.format
                if 'exit_price' in existing_cols:
                    formatters['exit_price'] = '{:.5f}'.format

                with pd.option_context('display.max_rows', None,
                                       'display.max_columns', None,
                                       'display.width', 1000):
                    trades_string = trades_df[existing_cols].to_string(index=False, formatters=formatters)
                results_file.write(trades_string + "\n")
            else:
                w("No trades to display.")

            logger.info(f"Successfully saved results to: {log_file_path}")
        except Exception as e:
            logger.critical(f"Failed to save results to {log_file_path}: {e}", exc_info=True)
            return None

        logger.info(f"--- Backtest for {file_suffix} completed ---")
        return summary_results

    finally:
        results_file.close()


def save_summary_report(all_results, output_folder):
    summary_file_path = os.path.join(output_folder, "summary_all_periods.txt")
    try:
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(" SUMMARY OF RESULTS FOR ALL PERIODS ".center(80, "=") + "\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of analyzed periods: {len(all_results)}\n\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Period':<30} {'Trades':<12} {'Winrate':<12} {'Profit/Loss':<15} {'Max DD':<12}\n")
            f.write("-" * 80 + "\n")

            total_all_trades = 0
            total_all_pnl = 0.0
            max_drawdown_overall = 0.0

            for result in all_results:
                f.write(
                    f"{result['period']:<30} {result['total_trades']:<12} "
                    f"{result['win_rate']:>10.2%} {result['total_pnl_pips']:>13.2f} "
                    f"{result['max_drawdown_pips']:>10.2f}\n"
                )
                total_all_trades += result['total_trades']
                total_all_pnl += result['total_pnl_pips']
                max_drawdown_overall = max(max_drawdown_overall, result['max_drawdown_pips'])

            f.write("-" * 80 + "\n")
            avg_winrate = sum(r['win_rate'] for r in all_results) / len(all_results) if all_results else 0
            f.write("\n" + "=" * 80 + "\n")
            f.write(" OVERALL SUMMARY ".center(80, "=") + "\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total number of trades:             {total_all_trades}\n")
            f.write(f"Average winrate across all periods: {avg_winrate:.2%}\n")
            f.write(f"Total profit/loss:                  {total_all_pnl:.2f} pips\n")
            f.write(f"Highest drawdown:                   {max_drawdown_overall:.2f} pips\n")
            f.write("=" * 80 + "\n")

        logger.info(f"Summary of all periods saved to: {summary_file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save summary: {e}")
        return False


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info(" Starting backtest series ".center(60, "="))
    logger.info("=" * 60)

    try:
        (trading_config, trend_criteria_config, INPUT_DATA_FOLDER, FILE_NAME,
         RESULTS_FOLDER, DATE_RANGES_TO_TEST, REQUIRED_COLUMNS, SIGNAL_COLUMN) = load_config(CONFIG_FILE_PATH)
    except SystemExit:
        sys.exit(1)

    results_folder_name = os.path.basename(RESULTS_FOLDER.rstrip("/\\"))
    RESULTS_FOLDER = os.path.join(OUTPUTS_DIR, results_folder_name)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    logger.info(f"Input data:      {INPUT_DATA_FOLDER}/{FILE_NAME}")
    logger.info(f"Results folder:  {RESULTS_FOLDER}")
    logger.info(f"Signal column:   {SIGNAL_COLUMN}")
    log_config(trading_config)

    input_file_path_full = os.path.join(INPUT_DATA_FOLDER, FILE_NAME)

    all_period_results = []

    for start_date, end_date, suffix in DATE_RANGES_TO_TEST:
        result = run_backtest(
            start_date, end_date, suffix,
            input_file_path_full, RESULTS_FOLDER,
            trading_config, trend_criteria_config,
            REQUIRED_COLUMNS, SIGNAL_COLUMN,
        )
        if result is not None:
            all_period_results.append(result)

    if all_period_results:
        save_summary_report(all_period_results, RESULTS_FOLDER)
        logger.info("=" * 60)
        logger.info(" All backtests completed ".center(60, "="))
        logger.info("=" * 60)
    else:
        logger.warning("No backtests could be executed.")