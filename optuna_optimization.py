# -*- coding: utf-8 -*-
import subprocess
import sys
import json
import optuna
import os
import re
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import numpy as np
from collections import defaultdict
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER

OPTUNA_CONFIG_PATH = "config_files/optuna_config.json"


def load_optuna_config() -> Dict[str, Any]:
    try:
        with open(OPTUNA_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for key in ["n_trials", "min_trades", "optimize_features", "optimize_hyperparameters"]:
            if key not in cfg:
                print(f"ERROR: Missing key '{key}' in {OPTUNA_CONFIG_PATH}")
                sys.exit(1)
        return cfg
    except FileNotFoundError:
        print(f"ERROR: {OPTUNA_CONFIG_PATH} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading {OPTUNA_CONFIG_PATH}: {e}")
        sys.exit(1)


OPTUNA_CFG           = load_optuna_config()
N_TRIALS             = OPTUNA_CFG["n_trials"]
MIN_TRADES           = OPTUNA_CFG["min_trades"]
OPTIMIZE_FEATURES    = OPTUNA_CFG["optimize_features"]
OPTIMIZE_HYPERPARAMS = OPTUNA_CFG["optimize_hyperparameters"]
SEARCH_SPACE         = OPTUNA_CFG.get("hyperparameter_search_space", {})

if OPTIMIZE_FEATURES and OPTIMIZE_HYPERPARAMS:
    OPT_MODE = "FEATURES + HYPERPARAMETERS"
elif OPTIMIZE_FEATURES:
    OPT_MODE = "FEATURES ONLY"
elif OPTIMIZE_HYPERPARAMS:
    OPT_MODE = "HYPERPARAMETERS ONLY"
else:
    print("ERROR: Both optimize_features and optimize_hyperparameters are false in optuna_config.json.")
    sys.exit(1)

OUTPUT_DIR = "outputs/optuna"


def load_features() -> List[str]:
    try:
        with open("features_lists/features_list.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            feat = data
        elif isinstance(data, dict) and "features" in data:
            feat = data["features"]
        else:
            raise ValueError("Invalid format in features_list.json")
        print(f"Loaded {len(feat)} features from features_list.json")
        return feat
    except FileNotFoundError:
        print("ERROR: features_list.json not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading features_list.json: {e}")
        sys.exit(1)


def discover_period_files(results_dir="outputs/results"):
    from pathlib import Path
    results_path = Path(results_dir)
    return [f.name for f in sorted(results_path.glob("log_results_*.txt"))] if results_path.exists() else []


def get_period_name(filename):
    from pathlib import Path
    return Path(filename).stem.replace("log_results_", "")


features = load_features()


class AdvancedFeatureTracker:

    def __init__(self):
        self.feature_scores = defaultdict(list)
        self.feature_pairs  = defaultdict(list)
        self.param_combos   = []
        self.best_configs   = []
        self.iteration      = 0
        self.ema_alpha      = 0.3

    def update(self, selected_features: List[str], params: Dict, winrate: float):
        self.iteration += 1
        time_weight = 1.0 + (self.iteration / 100) * 0.2
        for feature in selected_features:
            self.feature_scores[feature].append(winrate * time_weight)

        if winrate > 0.5:
            for i, f1 in enumerate(selected_features):
                for f2 in selected_features[i + 1:]:
                    self.feature_pairs[tuple(sorted([f1, f2]))].append(winrate)

        self.param_combos.append({'params': params.copy(), 'features': selected_features.copy(),
                                  'winrate': winrate, 'iteration': self.iteration})
        self.best_configs = sorted(self.param_combos, key=lambda x: x['winrate'], reverse=True)[:10]

    def get_smart_features(self, n: int = 35, exploration_rate: float = 0.2) -> List[str]:
        if not self.feature_scores or len(self.feature_scores) < 5:
            return list(np.random.choice(features, size=min(n, len(features)), replace=False))

        feature_ema_scores = {}
        for feature, scores in self.feature_scores.items():
            if scores:
                ema = scores[0]
                for score in scores[1:]:
                    ema = self.ema_alpha * score + (1 - self.ema_alpha) * ema
                stability_bonus = 1.0 / (1.0 + np.std(scores))
                good_usage = sum(1 for s in scores if s > 0.5) / max(len(scores), 1)
                feature_ema_scores[feature] = ema * (1 + 0.2 * stability_bonus + 0.3 * good_usage)

        synergy_boost = defaultdict(float)
        for pair, scores in self.feature_pairs.items():
            if len(scores) >= 3 and np.mean(scores) > 0.52:
                boost = (np.mean(scores) - 0.5) * 2
                for feat in pair:
                    synergy_boost[feat] += boost
        for feat in feature_ema_scores:
            feature_ema_scores[feat] += synergy_boost.get(feat, 0) * 0.15

        sorted_f = sorted(feature_ema_scores.items(), key=lambda x: x[1], reverse=True)
        n_exploit = int(n * (1 - exploration_rate))
        selected  = [f[0] for f in sorted_f[:n_exploit]]
        remaining = [f for f in features if f not in selected]
        untested  = [f for f in remaining if f not in self.feature_scores]
        n_explore = n - n_exploit
        pool = untested if (untested and len(untested) >= n_explore) else remaining
        selected.extend(list(np.random.choice(pool, size=min(n_explore, len(pool)), replace=False)))
        return selected[:n]

    def get_feature_groups(self) -> Dict[str, List[str]]:
        return {
            'trend':              [f for f in features if any(x in f.lower() for x in ['sma', 'ema', 'trend', 'macd'])],
            'momentum':           [f for f in features if any(x in f.lower() for x in ['rsi', 'mfi', 'roc', 'momentum'])],
            'volatility':         [f for f in features if any(x in f.lower() for x in ['atr', 'volatility', 'bb_'])],
            'volume':             [f for f in features if any(x in f.lower() for x in ['volume', 'vwap'])],
            'price_action':       [f for f in features if any(x in f.lower() for x in ['candle', 'shadow', 'body'])],
            'support_resistance': [f for f in features if any(x in f.lower() for x in ['resistance', 'support', 'pp', 'fib'])],
            'session':            [f for f in features if any(x in f.lower() for x in ['session', 'hour', 'london', 'ny'])],
            'lag':                [f for f in features if 'lag' in f.lower()],
        }

    def save_analysis(self, filepath: str = None):
        if filepath is None:
            filepath = f"{OUTPUT_DIR}/advanced_feature_analysis.json"
        groups = self.get_feature_groups()
        group_performance = {}
        for gname, gfeats in groups.items():
            scores = [s for f in gfeats for s in self.feature_scores.get(f, [])]
            if scores:
                group_performance[gname] = {
                    'avg_score': float(np.mean(scores)), 'median_score': float(np.median(scores)),
                    'std': float(np.std(scores)), 'count': len(scores)
                }
        top_pairs = [
            {'features': list(pair), 'avg_winrate': float(np.mean(scores)), 'count': len(scores)}
            for pair, scores in sorted(self.feature_pairs.items(),
                                       key=lambda x: np.mean(x[1]) if x[1] else 0, reverse=True)[:20]
            if scores
        ]
        analysis = {
            'iteration': self.iteration,
            'top_features': self.get_smart_features(40, exploration_rate=0),
            'group_performance': group_performance,
            'top_synergistic_pairs': top_pairs,
            'best_configurations': [
                {'winrate': c['winrate'], 'iteration': c['iteration'],
                 'n_features': len(c['features']), 'top_10_features': c['features'][:10]}
                for c in self.best_configs[:5]
            ],
            'feature_statistics': {
                feature: {
                    'ema_score':    float(np.mean(scores[-5:])) if len(scores) >= 5 else float(np.mean(scores)),
                    'avg_score':    float(np.mean(scores)),
                    'median_score': float(np.median(scores)),
                    'std':          float(np.std(scores)),
                    'usage_count':  len(scores),
                    'recent_trend': 'improving' if len(scores) >= 5 and np.mean(scores[-3:]) > np.mean(scores[:3]) else 'stable'
                }
                for feature, scores in self.feature_scores.items() if scores
            }
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)


feature_tracker = AdvancedFeatureTracker()


def run_script(script_path: str) -> bool:
    try:
        subprocess.run([sys.executable, script_path], check=True, capture_output=True, text=True)
        print(f"OK: {script_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error in {script_path}: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def read_winrate() -> float:
    validation_path = "outputs/results/log_optuna_validation.txt"
    try:
        with open(validation_path, "r", encoding="utf-8") as f:
            content = f.read()

        trades_match = re.search(r'Total number of trades:\s*(\d+)', content)
        if not trades_match:
            print(f"Warning: 'Total number of trades' not found in {validation_path} — returning winrate 0.0")
            return 0.0
        total_trades = int(trades_match.group(1))
        if total_trades < MIN_TRADES:
            print(f"Warning: Too few trades ({total_trades} < {MIN_TRADES} MIN_TRADES) — returning winrate 0.0")
            return 0.0

        wr_match = re.search(r'Profitable trades:\s*\d+\s*\((\d+\.\d+)%\)', content)
        if not wr_match:
            print(f"Warning: 'Profitable trades' percentage not found in {validation_path} — returning winrate 0.0")
            return 0.0
        return float(wr_match.group(1)) / 100.0

    except Exception as e:
        print(f"Warning: Error reading winrate: {e}")
        return 0.0


def read_monthly_results() -> Dict[str, Dict[str, Any]]:
    period_files = discover_period_files()
    monthly_data: Dict[str, Dict[str, Any]] = {}

    for filename in period_files:
        month_name = get_period_name(filename)
        filepath   = os.path.join("outputs/results", filename)
        try:
            if not os.path.exists(filepath):
                print(f"Warning: File {filepath} does not exist")
                continue
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            data = {'total_trades': 0, 'profitable_trades': 0,
                    'profitable_pct': 0.0, 'total_profit': 0.0, 'max_drawdown': 0.0}
            found = {'total_trades': False, 'profitable_trades': False,
                     'total_profit': False, 'max_drawdown': False}

            for line in content.split('\n'):
                line = line.strip()
                try:
                    if 'Total number of trades:' in line:
                        data['total_trades'] = int(line.split(':')[1].strip())
                        found['total_trades'] = True
                    elif 'Profitable trades:' in line:
                        parts = line.split(':')[1].strip().split('(')
                        data['profitable_trades'] = int(parts[0].strip())
                        data['profitable_pct']    = float(parts[1].replace('%)', '').strip())
                        found['profitable_trades'] = True
                    elif 'Total P/L:' in line:
                        data['total_profit'] = float(line.split(':')[1].replace('pips', '').strip())
                        found['total_profit'] = True
                    elif 'Maximum drawdown:' in line:
                        data['max_drawdown'] = float(line.split(':')[1].replace('pips', '').strip())
                        found['max_drawdown'] = True
                except (ValueError, IndexError) as e:
                    print(f"Warning: {month_name}: Error parsing '{line[:50]}': {e}")

            missing = [k for k, v in found.items() if not v]
            if missing:
                print(f"Warning: {month_name}: Values not found: {', '.join(missing)}")
            if data['total_trades'] == 0:
                print(f"Warning: {month_name}: No trades in file")

            monthly_data[month_name] = data
            print(f"OK: {month_name}: {data['total_trades']} trades, winrate: {data['profitable_pct']:.2f}%")

        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            monthly_data[month_name] = {
                'total_trades': 0, 'profitable_trades': 0,
                'profitable_pct': 0.0, 'total_profit': 0.0, 'max_drawdown': 0.0
            }

    print(f"\n{'=' * 60}")
    print(f"Read summary: {len(monthly_data)}/{len(period_files)} periods")
    print(f"Periods with data: {sum(1 for d in monthly_data.values() if d['total_trades'] > 0)}/{len(monthly_data)}")
    print(f"{'=' * 60}\n")
    return monthly_data


def _early_stopping_bounds(learning_rate: float) -> Tuple[int, int]:
    if learning_rate >= 0.15:
        return 20, 40
    elif learning_rate >= 0.05:
        return 40, 100
    else:
        return 100, 200


def suggest_hyperparameters(trial: optuna.Trial, base_params: Dict[str, Any]) -> Dict[str, Any]:
    params = base_params.copy()

    for param_name, spec in SEARCH_SPACE.items():
        kind    = spec["type"]
        use_log = spec.get("log", False)
        if kind == "int":
            value = trial.suggest_int(param_name, spec["low"], spec["high"])
        elif kind == "float":
            value = trial.suggest_float(param_name, spec["low"], spec["high"], log=use_log)
        else:
            continue
        params[param_name] = value

    lr = params.get("learning_rate", 0.1)
    es_low, es_high = _early_stopping_bounds(lr)
    params["early_stopping_rounds_param"] = trial.suggest_int(
        "early_stopping_rounds", es_low, es_high
    )
    print(f"  early_stopping_rounds: {params['early_stopping_rounds_param']} "
          f"(lr={lr:.4f}, range=[{es_low}, {es_high}])")

    return params


_META_KEYS = {"max_features_count", "n_features_to_select"}


def update_model_config(params: Dict[str, Any], selected_features: List[str]) -> None:
    config_path = "config_files/model_config.json"
    if not os.path.exists(config_path) or os.path.getsize(config_path) == 0:
        raise FileNotFoundError(f"{config_path} not found or empty")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["model"]["features"] = selected_features

    for k, v in params.items():
        if k == "window_size":
            config["model"]["window_size"] = v
        elif k == "early_stopping_rounds_param":
            config["model"]["early_stopping_rounds"] = v
        elif k in _META_KEYS:
            continue
        else:
            config["model"]["params"][k] = v

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)


def load_existing_config() -> Tuple[Dict[str, Any], List[str]]:
    config_path = "config_files/model_config.json"
    if not os.path.exists(config_path):
        print("ERROR: model_config.json not found!")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model_cfg = config.get("model", {})
    params    = model_cfg.get("params", {}).copy()
    existing_features = model_cfg.get("features", [])
    params["window_size"]                 = model_cfg.get("window_size", 20)
    params["early_stopping_rounds_param"] = model_cfg.get("early_stopping_rounds", 40)

    print(f"Loaded config: {len(existing_features)} features, {len(params)} parameters")
    print("MODEL PARAMETERS (baseline):")
    for key in ["max_depth", "learning_rate", "window_size", "gamma", "reg_alpha", "reg_lambda",
                "early_stopping_rounds_param"]:
        print(f"   {key}: {params.get(key)}")
    print()
    return params, existing_features


def log_iteration(trial_number: int, selected_features: List[str],
                  winrate: float, best_so_far: float, improvement: float) -> None:
    os.makedirs("logs", exist_ok=True)
    timestamp      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_improvement = winrate > best_so_far
    with open(os.path.join("logs", "optuna_optimization.txt"), "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"Trial #{trial_number} | {timestamp} | Mode: {OPT_MODE}\n")
        f.write(f"Winrate: {winrate:.4f}"
                f"{' (+' + str(round(improvement * 100, 2)) + '%)' if is_improvement else ''}\n")
        f.write(f"Best so far: {best_so_far:.4f}\n")
        f.write(f"Number of features: {len(selected_features)}\n")
        f.write(f"Selected features:\n  {', '.join(selected_features)}\n")


def save_trial_config(trial_number: int, params: Dict[str, Any],
                      selected_features: List[str], winrate: float,
                      monthly_results: Dict[str, Dict[str, Any]]) -> None:
    trial_dir = f"{OUTPUT_DIR}/trials_history"
    os.makedirs(trial_dir, exist_ok=True)
    trial_data = {
        "trial_number":      trial_number,
        "timestamp":         datetime.now().isoformat(),
        "optimization_mode": OPT_MODE,
        "winrate_overall":   winrate,
        "hyperparameters":   {k: v for k, v in params.items() if k not in _META_KEYS},
        "model_config": {
            "window_size":                params.get("window_size", 20),
            "early_stopping_rounds_param": params.get("early_stopping_rounds_param", 40),
            "n_features":                 len(selected_features),
        },
        "features":        selected_features,
        "monthly_results": monthly_results,
    }
    filepath = os.path.join(trial_dir, f"trial_{trial_number:04d}.json")
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trial_data, f, indent=2, ensure_ascii=False)
        print(f"Saved trial: {filepath}")
    except Exception as e:
        print(f"Error saving trial: {e}")


def save_best_config(winrate: float, params: Dict[str, Any],
                     selected_features: List[str], trial_number: int) -> None:
    temp_path  = f"{OUTPUT_DIR}/model_best_config_temp.json"
    final_path = f"{OUTPUT_DIR}/model_best_config.json"
    current_best = 0.0
    if os.path.exists(final_path):
        try:
            with open(final_path, "r", encoding="utf-8") as f:
                current_best = json.load(f).get("best_winrate", 0.0)
        except Exception:
            pass
    if winrate <= current_best:
        return
    config = {
        "best_winrate":      winrate,
        "trial_number":      trial_number,
        "optimization_mode": OPT_MODE,
        "timestamp":         datetime.now().isoformat(),
        "model": {
            "params": {
                k: v for k, v in params.items()
                if k not in _META_KEYS and k not in ("window_size", "early_stopping_rounds_param")
            },
            "window_size":                params.get("window_size", 20),
            "early_stopping_rounds_param": params.get("early_stopping_rounds_param", 40),
            "features":                   selected_features,
        },
        "statistics": {
            "n_features":                len(selected_features),
            "improvement_from_previous": float(winrate - current_best) if current_best > 0 else None,
        }
    }
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)
        import shutil
        shutil.move(temp_path, final_path)
        improvement = (winrate - current_best) * 100 if current_best > 0 else winrate * 100
        print(f"\nNEW RECORD! Winrate: {winrate:.4f} (+{improvement:.2f}%)")
        print(f"   Saved to: {final_path}\n")
    except Exception as e:
        print(f"Error saving best config: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def load_all_trials() -> List[Dict[str, Any]]:
    trials_dir = f"{OUTPUT_DIR}/trials_history"
    if not os.path.exists(trials_dir):
        return []
    result = []
    for filename in sorted(os.listdir(trials_dir)):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(trials_dir, filename), 'r', encoding='utf-8') as f:
                    result.append(json.load(f))
            except Exception as e:
                print(f"Warning: Error loading {filename}: {e}")
    return result


def _base_styles():
    base = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=base['Heading1'], fontSize=20,
                                 textColor=colors.HexColor('#1f4788'), spaceAfter=20, alignment=TA_CENTER)
    heading_style = ParagraphStyle('TrialHeading', parent=base['Heading2'], fontSize=14,
                                   textColor=colors.HexColor('#2c5aa0'), spaceAfter=8, spaceBefore=15)
    return base, title_style, heading_style


def _build_period_table(monthly: Dict, months: List[str]) -> Table:
    rows = [['Period', 'Trades', 'Winrate', 'Profit/Loss', 'Max DD'],
            ['_' * 20, '_' * 12, '_' * 12, '_' * 12, '_' * 12]]
    profit_rows, loss_rows = [], []

    for row_idx, month in enumerate(months, start=2):
        d      = monthly.get(month, {})
        trades = d.get('total_trades', 0)
        profit = d.get('total_profit', 0)
        rows.append([
            f"results_{month}",
            str(trades) if trades > 0 else '-',
            f"{d.get('profitable_pct', 0):.2f}%" if d.get('profitable_pct', 0) > 0 else '-',
            f"{profit:.2f}"                       if profit != 0 else '-',
            f"{d.get('max_drawdown', 0):.2f}"    if d.get('max_drawdown', 0) != 0 else '-',
        ])
        if trades > 0:
            (profit_rows if profit > 0 else loss_rows if profit < 0 else []).append(row_idx)

    style = TableStyle([
        ('FONTNAME',       (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',       (0, 0), (-1, 0),  10),
        ('ALIGN',          (0, 0), (0, -1),  'LEFT'),
        ('ALIGN',          (1, 0), (-1, -1), 'RIGHT'),
        ('LINEBELOW',      (0, 1), (-1, 1),  1, colors.black),
        ('FONTNAME',       (0, 2), (-1, -1), 'Helvetica'),
        ('FONTSIZE',       (0, 2), (-1, -1), 9),
        ('TOPPADDING',     (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING',  (0, 0), (-1, -1), 4),
        ('LEFTPADDING',    (0, 0), (-1, -1), 6),
        ('RIGHTPADDING',   (0, 0), (-1, -1), 6),
    ])
    for r in profit_rows:
        style.add('TEXTCOLOR', (3, r), (3, r), colors.green)
        style.add('FONTNAME',  (3, r), (3, r), 'Helvetica-Bold')
    for r in loss_rows:
        style.add('TEXTCOLOR', (3, r), (3, r), colors.red)
        style.add('FONTNAME',  (3, r), (3, r), 'Helvetica-Bold')

    table = Table(rows, colWidths=[4.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
    table.setStyle(style)
    return table


def _make_doc(path: str) -> SimpleDocTemplate:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return SimpleDocTemplate(path, pagesize=A4,
                             rightMargin=2*cm, leftMargin=2*cm,
                             topMargin=2*cm,   bottomMargin=2*cm)


def create_pdf_report(trials_data: List[Dict[str, Any]], output_path: str = None):
    if output_path is None:
        output_path = f"{OUTPUT_DIR}/trials_summary.pdf"
    doc = _make_doc(output_path)
    base, title_style, heading_style = _base_styles()
    months   = [get_period_name(f) for f in discover_period_files()]
    elements = [Paragraph("Optimization Report - All Trials", title_style), Spacer(1, 0.5*cm)]

    if trials_data:
        elements.append(Paragraph(
            f"<b>Summary:</b><br/>Total trials: {len(trials_data)}<br/>"
            f"Optimization mode: {OPT_MODE}<br/>"
            f"Best winrate: {max(t['winrate_overall'] for t in trials_data):.4f}<br/>"
            f"Average winrate: {np.mean([t['winrate_overall'] for t in trials_data]):.4f}<br/>"
            f"Report date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            base['Normal']))
        elements.append(Spacer(1, 0.8*cm))

    for i, trial in enumerate(sorted(trials_data, key=lambda x: x['trial_number'])):
        mode_label = trial.get('optimization_mode', OPT_MODE)
        elements.append(Paragraph(f"<b>Trial {trial['trial_number']}:</b> [Mode: {mode_label}]", heading_style))
        elements.append(Paragraph("_" * 80, base['Normal']))
        elements.append(Spacer(1, 0.2*cm))
        elements.append(_build_period_table(trial.get('monthly_results', {}), months))
        elements.append(Spacer(1, 0.8*cm))
        if i < len(trials_data) - 1:
            elements.append(PageBreak())

    try:
        doc.build(elements)
        print(f"\nPDF report created: {output_path}")
    except Exception as e:
        print(f"\nError creating PDF: {e}")


def create_profitable_trials_pdf(trials_data: List[Dict[str, Any]], output_path: str = None):
    if output_path is None:
        output_path = f"{OUTPUT_DIR}/profitable_trials.pdf"
    months = [get_period_name(f) for f in discover_period_files()]

    profitable = [
        t for t in trials_data
        if sum(t.get('monthly_results', {}).get(m, {}).get('total_trades', 0) for m in months) > 0
        and all(
            t.get('monthly_results', {}).get(m, {}).get('total_profit', 0) > 0
            for m in months
            if t.get('monthly_results', {}).get(m, {}).get('total_trades', 0) > 0
        )
    ]

    print(f"\n{'=' * 60}")
    print(f"Found {len(profitable)} trials profitable in every period")
    print(f"{'=' * 60}\n")
    if not profitable:
        print("Warning: No trials meet the criteria - PDF not created")
        return

    doc = _make_doc(output_path)
    base, title_style, heading_style = _base_styles()
    elements = [
        Paragraph("Report - Trials Profitable in All Periods", title_style),
        Spacer(1, 0.5*cm),
        Paragraph(
            f"<b>Summary:</b><br/>Trials meeting criteria: {len(profitable)}<br/>"
            f"Optimization mode: {OPT_MODE}<br/>"
            f"Best winrate: {max(t['winrate_overall'] for t in profitable):.4f}<br/>"
            f"Average winrate: {np.mean([t['winrate_overall'] for t in profitable]):.4f}<br/>"
            f"Report date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            base['Normal']),
        Spacer(1, 0.8*cm),
    ]
    sorted_profitable = sorted(profitable, key=lambda x: x['trial_number'])
    for i, trial in enumerate(sorted_profitable):
        mode_label = trial.get('optimization_mode', OPT_MODE)
        elements.append(Paragraph(
            f"<b>Trial {trial['trial_number']}:</b> "
            f"(Winrate: {trial['winrate_overall']:.4f}, Mode: {mode_label})", heading_style))
        elements.append(Paragraph("_" * 80, base['Normal']))
        elements.append(Spacer(1, 0.2*cm))
        elements.append(_build_period_table(trial.get('monthly_results', {}), months))
        elements.append(Spacer(1, 0.8*cm))
        if i < len(sorted_profitable) - 1:
            elements.append(PageBreak())

    try:
        doc.build(elements)
        print(f"\nPDF report (profitable trials) created: {output_path}")
    except Exception as e:
        print(f"\nError creating PDF: {e}")


def create_minimum_losses_pdf(trials_data: List[Dict[str, Any]], output_path: str = None):
    if output_path is None:
        output_path = f"{OUTPUT_DIR}/minimum_losses_trials.pdf"
    months = [get_period_name(f) for f in discover_period_files()]

    scored = []
    for trial in trials_data:
        monthly      = trial.get('monthly_results', {})
        total_trades = sum(monthly.get(m, {}).get('total_trades', 0) for m in months)
        if total_trades == 0:
            continue
        loss_count = sum(
            1 for m in months
            if monthly.get(m, {}).get('total_trades', 0) > 0
            and monthly.get(m, {}).get('total_profit', 0) < 0
        )
        scored.append({'trial': trial, 'loss_count': loss_count, 'total_trades': total_trades})

    if not scored:
        print("Warning: No trials with trades - PDF not created")
        return

    min_losses = min(s['loss_count'] for s in scored)
    selected   = [s for s in scored if s['loss_count'] == min_losses]
    print(f"\n{'=' * 60}")
    print(f"Minimum number of loss periods: {min_losses}")
    print(f"Found {len(selected)} trials with {min_losses} loss periods")
    print(f"{'=' * 60}\n")

    doc = _make_doc(output_path)
    base, title_style, heading_style = _base_styles()
    elements = [
        Paragraph(f"Report - Trials with Fewest Loss Periods ({min_losses})", title_style),
        Spacer(1, 0.5*cm),
        Paragraph(
            f"<b>Summary:</b><br/>Trials meeting criteria: {len(selected)}<br/>"
            f"Optimization mode: {OPT_MODE}<br/>"
            f"Minimum number of loss periods: {min_losses}<br/>"
            f"Best winrate in group: {max(s['trial']['winrate_overall'] for s in selected):.4f}<br/>"
            f"Average winrate in group: {np.mean([s['trial']['winrate_overall'] for s in selected]):.4f}<br/>"
            f"Report date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            base['Normal']),
        Spacer(1, 0.8*cm),
    ]
    sorted_selected = sorted(selected, key=lambda x: x['trial']['winrate_overall'], reverse=True)
    for idx, st in enumerate(sorted_selected):
        trial      = st['trial']
        mode_label = trial.get('optimization_mode', OPT_MODE)
        elements.append(Paragraph(
            f"<b>Trial {trial['trial_number']}:</b> "
            f"(Winrate: {trial['winrate_overall']:.4f}, "
            f"Loss periods: {st['loss_count']}/{len(months)}, Mode: {mode_label})", heading_style))
        elements.append(Paragraph("_" * 80, base['Normal']))
        elements.append(Spacer(1, 0.2*cm))
        elements.append(_build_period_table(trial.get('monthly_results', {}), months))
        elements.append(Spacer(1, 0.8*cm))
        if idx < len(sorted_selected) - 1:
            elements.append(PageBreak())

    try:
        doc.build(elements)
        print(f"\nPDF report (minimum losses) created: {output_path}")
    except Exception as e:
        print(f"\nError creating PDF: {e}")


def objective(trial: optuna.Trial) -> float:
    base_params, existing_features = load_existing_config()

    if OPTIMIZE_FEATURES:
        if trial.number == 0:
            selected_features = existing_features
            feat_phase = "BASELINE"
        elif trial.number < 10:
            selected_features = list(np.random.choice(features, size=20, replace=False))
            feat_phase = "EXPLORATION"
        elif trial.number < 30:
            selected_features = feature_tracker.get_smart_features(22, exploration_rate=0.3)
            feat_phase = "LEARNING"
        else:
            selected_features = feature_tracker.get_smart_features(25, exploration_rate=0.15)
            feat_phase = "EXPLOITATION"
    else:
        selected_features = existing_features
        feat_phase = "FIXED"

    if not selected_features:
        return 0.0

    if OPTIMIZE_HYPERPARAMS:
        params   = suggest_hyperparameters(trial, base_params)
        hp_phase = "SAMPLED"
    else:
        params   = base_params.copy()
        hp_phase = "FIXED"

    params["max_features_count"] = len(selected_features)

    print(f"\n{'=' * 80}")
    print(f"Trial #{trial.number} | Mode: {OPT_MODE}")
    print(f"  Features ({len(selected_features)}): [{feat_phase}]")
    print(f"  Hyperparameters:           [{hp_phase}]")
    print(f"{'=' * 80}")

    try:
        update_model_config(params, selected_features)
    except Exception as e:
        print(f"Config error: {e}")
        return 0.0

    for script in ["05_train_model.py", "06_generate_predictions.py", "07_backtest.py"]:
        if not run_script(script):
            return 0.0

    winrate         = read_winrate()
    monthly_results = read_monthly_results()
    save_trial_config(trial.number, params, selected_features, winrate, monthly_results)

    if OPTIMIZE_FEATURES:
        feature_tracker.update(selected_features, params, winrate)

    try:
        completed  = [t for t in trial.study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE and t.value]
        best_so_far = max(t.value for t in completed) if completed else 0.0
        improvement = (winrate - best_so_far) / max(best_so_far, 0.01) if best_so_far > 0 else 0.0
    except Exception:
        best_so_far = 0.0
        improvement = 0.0

    log_iteration(trial.number, selected_features, winrate, best_so_far, improvement)
    save_best_config(winrate, params, selected_features, trial.number)

    print(f"\n{'=' * 80}")
    print(f"Winrate: {winrate:.4f} | Best so far: {best_so_far:.4f}")
    return winrate


def print_optimization_summary(study: optuna.Study):
    print(f"\n{'=' * 80}")
    print(f"OPTIMIZATION SUMMARY | Mode: {OPT_MODE}")
    print(f"{'=' * 80}\n")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("No completed trials.")
        return

    best = study.best_trial
    print(f"Best Winrate: {best.value:.4f} (Trial #{best.number})\n")

    values = [t.value for t in completed if t.value]
    if len(values) >= 10:
        print("PROGRESS:")
        print("-" * 80)
        for label, vals in [('First 10', values[:10]), ('Trials 11-30', values[10:30]), ('Last 10', values[-10:])]:
            if vals:
                print(f"  {label:15s}: Avg={np.mean(vals):.4f}, Max={np.max(vals):.4f}")

    if OPTIMIZE_FEATURES:
        print(f"\nFEATURE GROUP PERFORMANCE:")
        print("-" * 80)
        group_perf = []
        for name, feats in feature_tracker.get_feature_groups().items():
            scores = [s for f in feats for s in feature_tracker.feature_scores.get(f, [])]
            if scores:
                group_perf.append((name, np.mean(scores), len(scores)))
        for name, score, count in sorted(group_perf, key=lambda x: x[1], reverse=True):
            print(f"  {name:20s}: {score:.4f} (uses: {count})")

        print(f"\nTOP 20 FEATURES:")
        print("-" * 80)
        for i, feat in enumerate(feature_tracker.get_smart_features(20, exploration_rate=0), 1):
            stats = feature_tracker.feature_scores.get(feat, [])
            if stats:
                print(f"  {i:2d}. {feat:35s} | Score: {np.mean(stats):.4f}")

    print(f"\n{'=' * 80}\n")


def main():
    print(f"\n{'=' * 80}")
    print(f"OPTUNA OPTIMIZATION | Mode: {OPT_MODE}")
    print(f"{'=' * 80}\n")

    os.makedirs("logs", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/trials_history", exist_ok=True)

    existing_params, existing_features = load_existing_config()

    print(f"Settings (from {OPTUNA_CONFIG_PATH}):")
    print(f"   Optimize features:        {OPTIMIZE_FEATURES}")
    print(f"   Optimize hyperparameters: {OPTIMIZE_HYPERPARAMS}")
    print(f"   N_TRIALS:                 {N_TRIALS}")
    print(f"   MIN_TRADES filter:        {MIN_TRADES}")
    print(f"   Output directory:         {OUTPUT_DIR}/\n")

    sampler = (optuna.samplers.TPESampler(seed=42)
               if OPTIMIZE_HYPERPARAMS
               else optuna.samplers.RandomSampler(seed=42))

    db_path = f"sqlite:///{OUTPUT_DIR}/optuna_study.db"
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"XGBoost_{OPT_MODE.replace(' ', '_').replace('+', 'and')}",
        storage=db_path,
        load_if_exists=True,
    )

    print(f"Database: {db_path}")
    print(f"Completed trials so far: {len(study.trials)}\n")

    completed_so_far = [t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE and t.value]
    if completed_so_far:
        best = max(t.value for t in completed_so_far)
        print(f"Best result so far: {best:.4f}")
        if best >= 0.55:
            print("TARGET 55% already reached!\n")

    if OPTIMIZE_FEATURES:
        print("FEATURE STRATEGY:")
        print("  Trial 0:       Baseline (features from model_config.json)")
        print("  Trials 1-9:    Wide exploration  (20 random features)")
        print("  Trials 10-29:  Directed learning (22 smart features, 30% explore)")
        print("  Trials 30+:    Exploitation      (25 smart features, 15% explore)")

    if OPTIMIZE_HYPERPARAMS:
        print("HYPERPARAMETER STRATEGY:")
        print("  Sampler: TPE (Tree-structured Parzen Estimator)")
        print(f"  Parameters: {', '.join(SEARCH_SPACE.keys())}")
        print("  early_stopping_rounds: dynamic range based on learning_rate")
        print("    lr >= 0.15:  [20,  40]")
        print("    lr >= 0.05: [40, 100]")
        print("    lr <  0.05:  [100, 200]")

    print(f"\n{'=' * 80}\n")

    class EarlyStoppingCallback:
        def __init__(self, target=0.55, patience=30):
            self.target               = target
            self.patience             = patience
            self.no_improvement_count = 0
            self.best_value           = 0.0

        def __call__(self, study, trial):
            if trial.number % 5 == 0:
                if OPTIMIZE_FEATURES:
                    feature_tracker.save_analysis()
                trials_data = load_all_trials()
                if trials_data:
                    create_pdf_report(trials_data)
                    create_profitable_trials_pdf(trials_data)
                    create_minimum_losses_pdf(trials_data)

            if trial.value and trial.value > self.best_value:
                self.best_value           = trial.value
                self.no_improvement_count = 0
                if self.best_value >= self.target:
                    print(f"\nTARGET REACHED: {self.best_value:.4f} >= {self.target:.4f}")
                    print("Stopping optimization...\n")
                    study.stop()
            else:
                self.no_improvement_count += 1
                if self.no_improvement_count >= self.patience and trial.number > 50:
                    print(f"\nNo improvement for {self.patience} trials (best: {self.best_value:.4f})")
                    print("Consider stopping or analyzing results...\n")

    try:
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            timeout=72000,
            show_progress_bar=True,
            callbacks=[EarlyStoppingCallback(target=0.55, patience=30)],
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user")

    if OPTIMIZE_FEATURES:
        feature_tracker.save_analysis()

    trials_data = load_all_trials()
    if trials_data:
        create_pdf_report(trials_data)
        create_minimum_losses_pdf(trials_data)
        print(f"\nFinal PDF report contains {len(trials_data)} trials")

    print_optimization_summary(study)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        print("Results saved to:")
        print(f"  {OUTPUT_DIR}/model_best_config.json")
        print(f"  {OUTPUT_DIR}/trials_summary.pdf")
        print(f"  {OUTPUT_DIR}/trials_history/  (JSON per trial)")
        if OPTIMIZE_FEATURES:
            print(f"  {OUTPUT_DIR}/advanced_feature_analysis.json")
        print(f"  logs/optuna_optimization.txt\n")
    else:
        print("\nNo completed trials.\n")

    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()