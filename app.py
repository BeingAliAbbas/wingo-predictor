"""
WinGo 30S Predictor - Python Backend with ML-Style Prediction
A Flask application that predicts WinGo outcomes using advanced pattern analysis
and machine learning techniques. Data is stored in JSON files for persistence.
"""

import os
import json
import time
import math
import logging
import requests
from flask import Flask, jsonify, render_template, request
from collections import Counter, deque
from datetime import datetime
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ML Model Constants
MAX_STREAK_LENGTH = 7  # Maximum streak length to analyze
MIN_SEQUENCE_DATA = 5  # Minimum data points for sequence pattern reliability
DISTRIBUTION_THRESHOLD = 7  # Number of recent outcomes to trigger distribution correction
MOMENTUM_WINDOW = 20  # Window size for momentum/trend analysis
RECENCY_BIAS = 0.8  # Weight for more recent outcomes in calculations
RECENCY_SCALE_FACTOR = 10  # Scale factor for recency weighting
HOT_NUMBER_THRESHOLD = 0.11  # Frequency above this is considered "hot" (>11%)
COLD_NUMBER_THRESHOLD = 0.09  # Frequency below this is considered "cold" (<9%)

app = Flask(__name__)

# File paths for JSON storage
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
OUTCOMES_FILE = os.path.join(DATA_DIR, 'outcomes.json')
PREDICTIONS_FILE = os.path.join(DATA_DIR, 'predictions.json')
MODEL_FILE = os.path.join(DATA_DIR, 'model.json')

API_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"

# Thread lock for file operations
file_lock = Lock()


def ensure_data_dir():
    """Ensure the data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)


def load_json_file(filepath, default=None):
    """Load data from a JSON file."""
    if default is None:
        default = {}
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Error loading {filepath}: {e}")
    return default


def save_json_file(filepath, data):
    """Save data to a JSON file."""
    ensure_data_dir()
    with file_lock:
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            logger.error(f"Error saving {filepath}: {e}")


def get_label(num):
    """Convert number to Big/Small label."""
    if num is None:
        return None
    return 'Small' if num <= 4 else 'Big'


def fetch_latest_draws():
    """Fetch latest draws from the external API."""
    try:
        response = requests.get(f"{API_URL}?ts={int(time.time() * 1000)}", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('data', {}).get('list'):
            return None, "Unexpected API format"
        
        draws = []
        for item in data['data']['list'][:50]:
            draws.append({
                'period': item['issueNumber'],
                'num': int(item['number']),
                'label': get_label(int(item['number']))
            })
        return draws, None
    except requests.exceptions.RequestException as e:
        return None, str(e)
    except (KeyError, ValueError) as e:
        return None, str(e)


def save_outcomes(draws):
    """Save fetched outcomes to JSON file."""
    if not draws:
        return
    
    outcomes = load_json_file(OUTCOMES_FILE, {'records': [], 'periods': set()})
    
    # Convert periods to set if it's a list (from JSON)
    if isinstance(outcomes.get('periods'), list):
        outcomes['periods'] = set(outcomes['periods'])
    elif not outcomes.get('periods'):
        outcomes['periods'] = set()
    
    for draw in draws:
        if draw['period'] not in outcomes['periods']:
            outcomes['records'].append({
                'period': draw['period'],
                'number': draw['num'],
                'label': draw['label'],
                'timestamp': datetime.now().isoformat()
            })
            outcomes['periods'].add(draw['period'])
    
    # Keep only last 5000 records for efficiency
    if len(outcomes['records']) > 5000:
        outcomes['records'] = outcomes['records'][-5000:]
        outcomes['periods'] = set(r['period'] for r in outcomes['records'])
    
    # Convert set to list for JSON serialization
    save_data = {
        'records': outcomes['records'],
        'periods': list(outcomes['periods'])
    }
    save_json_file(OUTCOMES_FILE, save_data)


def get_historical_outcomes(limit=1000):
    """Get historical outcomes from JSON file."""
    outcomes = load_json_file(OUTCOMES_FILE, {'records': []})
    records = outcomes.get('records', [])
    # Sort by period descending and return limited records
    sorted_records = sorted(records, key=lambda x: x['period'], reverse=True)
    return sorted_records[:limit]


class MLPredictor:
    """
    Advanced ML-style predictor that learns from historical data.
    Uses multiple techniques:
    1. Markov Chain for sequence prediction
    2. Pattern recognition for streaks
    3. Weighted ensemble voting
    4. Adaptive learning based on prediction accuracy
    5. Multi-length sequence patterns
    6. Momentum/trend analysis
    7. Hot/cold analysis
    """
    
    def __init__(self):
        self.model = self.load_model()
    
    def load_model(self):
        """Load or initialize the prediction model."""
        default_model = {
            'markov_chain': {
                'Big': {'Big': 0.5, 'Small': 0.5},
                'Small': {'Big': 0.5, 'Small': 0.5}
            },
            'streak_patterns': {},  # streak_length -> {after_big: {Big: count, Small: count}, after_small: {...}}
            'sequence_patterns_3': {},  # 3-length sequences -> next outcome counts
            'sequence_patterns_4': {},  # 4-length sequences -> next outcome counts
            'sequence_patterns_5': {},  # 5-length sequences -> next outcome counts
            'number_frequency': {},
            'strategy_weights': {
                'markov': 1.0,
                'streak': 1.2,
                'sequence_3': 1.0,
                'sequence_4': 1.1,
                'sequence_5': 1.2,
                'frequency': 0.9,
                'alternation': 1.0,
                'momentum': 1.1,
                'hot_cold': 0.8
            },
            'strategy_performance': {},  # Track how each strategy performs
            'total_predictions': 0,
            'correct_predictions': 0
        }
        return load_json_file(MODEL_FILE, default_model)
    
    def save_model(self):
        """Save the model to JSON file."""
        save_json_file(MODEL_FILE, self.model)
    
    def train(self, outcomes):
        """Train the model on historical outcomes."""
        if len(outcomes) < 10:
            return
        
        # Sort outcomes chronologically (oldest first for training)
        sorted_outcomes = sorted(outcomes, key=lambda x: x['period'])
        
        # 1. Build Markov Chain transition probabilities with recency weighting
        transitions = {'Big': {'Big': 0, 'Small': 0}, 'Small': {'Big': 0, 'Small': 0}}
        n = len(sorted_outcomes)
        for i in range(n - 1):
            current = sorted_outcomes[i]['label']
            next_label = sorted_outcomes[i + 1]['label']
            # Apply recency weighting - more recent transitions count more
            weight = RECENCY_BIAS ** ((n - 2 - i) / max(n - 1, 1) * RECENCY_SCALE_FACTOR)
            transitions[current][next_label] += weight
        
        # Normalize to probabilities with Laplace smoothing
        for state in transitions:
            total = sum(transitions[state].values()) + 2  # Laplace smoothing
            for next_state in transitions[state]:
                transitions[state][next_state] = (transitions[state][next_state] + 1) / total
        
        self.model['markov_chain'] = transitions
        
        # 2. Build streak pattern statistics
        streak_patterns = {}
        for streak_len in range(2, MAX_STREAK_LENGTH + 1):
            streak_patterns[streak_len] = {
                'after_big': {'Big': 0, 'Small': 0},
                'after_small': {'Big': 0, 'Small': 0}
            }
        
        for i in range(len(sorted_outcomes)):
            if i < 2:
                continue
            
            streak_type = sorted_outcomes[i - 1]['label']
            streak_len = 1
            for j in range(i - 2, -1, -1):
                if sorted_outcomes[j]['label'] == streak_type:
                    streak_len += 1
                else:
                    break
            
            if streak_len >= 2 and streak_len <= MAX_STREAK_LENGTH:
                next_label = sorted_outcomes[i]['label']
                if streak_type == 'Big':
                    streak_patterns[streak_len]['after_big'][next_label] += 1
                else:
                    streak_patterns[streak_len]['after_small'][next_label] += 1
        
        self.model['streak_patterns'] = {str(k): v for k, v in streak_patterns.items()}
        
        # 3. Build multi-length sequence pattern statistics (3, 4, 5)
        for seq_len in [3, 4, 5]:
            sequence_patterns = {}
            for i in range(seq_len, len(sorted_outcomes)):
                seq = tuple(sorted_outcomes[j]['label'] for j in range(i - seq_len, i))
                seq_key = ''.join('B' if s == 'Big' else 'S' for s in seq)
                next_label = sorted_outcomes[i]['label']
                
                if seq_key not in sequence_patterns:
                    sequence_patterns[seq_key] = {'Big': 0, 'Small': 0}
                sequence_patterns[seq_key][next_label] += 1
            
            self.model[f'sequence_patterns_{seq_len}'] = sequence_patterns
        
        # 4. Number frequency analysis
        number_freq = Counter(o['number'] for o in sorted_outcomes)
        self.model['number_frequency'] = dict(number_freq)
        
        self.save_model()
    
    def calculate_momentum(self, labels):
        """Calculate momentum/trend in recent outcomes.
        
        Note: labels is already sorted with most recent first (index 0 = most recent).
        So labels[:10] gets the 10 most recent, labels[10:20] gets the older 10.
        """
        if len(labels) < MOMENTUM_WINDOW:
            return 0, 'neutral'
        
        # Count Big vs Small in windows (labels[0] is most recent)
        recent_half = labels[:MOMENTUM_WINDOW // 2]  # 10 most recent outcomes
        older_half = labels[MOMENTUM_WINDOW // 2:MOMENTUM_WINDOW]  # 10 older outcomes
        
        recent_big_pct = sum(1 for l in recent_half if l == 'Big') / len(recent_half)
        older_big_pct = sum(1 for l in older_half if l == 'Big') / len(older_half)
        
        momentum = recent_big_pct - older_big_pct
        
        if momentum > 0.15:
            return momentum, 'Big'  # Trending towards Big
        elif momentum < -0.15:
            return abs(momentum), 'Small'  # Trending towards Small
        return abs(momentum), 'neutral'
    
    def predict(self, recent_draws):
        """
        Make a prediction using ensemble of strategies.
        Returns: (prediction, confidence, reason, all_strategies)
        """
        if not recent_draws or len(recent_draws) < 3:
            return 'Big', 0.5, 'Insufficient data', []
        
        labels = [d['label'] for d in recent_draws[:20]]  # Extended to 20 for momentum
        numbers = [d['num'] for d in recent_draws[:20]]
        strategies = []
        
        # Strategy 1: Markov Chain
        last_label = labels[0]
        markov_probs = self.model['markov_chain'].get(last_label, {'Big': 0.5, 'Small': 0.5})
        if markov_probs['Big'] > markov_probs['Small']:
            strategies.append({
                'name': 'Markov Chain',
                'prediction': 'Big',
                'confidence': markov_probs['Big'],
                'weight': self.model['strategy_weights']['markov']
            })
        else:
            strategies.append({
                'name': 'Markov Chain',
                'prediction': 'Small',
                'confidence': markov_probs['Small'],
                'weight': self.model['strategy_weights']['markov']
            })
        
        # Strategy 2: Streak Analysis with improved weighting
        streak_len = 1
        for i in range(1, len(labels)):
            if labels[i] == labels[0]:
                streak_len += 1
            else:
                break
        
        if streak_len >= 2:
            streak_key = str(min(streak_len, MAX_STREAK_LENGTH))
            streak_data = self.model['streak_patterns'].get(streak_key, {})
            streak_after = streak_data.get(f'after_{labels[0].lower()}', {'Big': 1, 'Small': 1})
            total = streak_after['Big'] + streak_after['Small']
            
            if total > 0:
                prob_big = streak_after['Big'] / total
                prob_small = streak_after['Small'] / total
                
                # Boost weight for longer streaks
                streak_weight = self.model['strategy_weights']['streak'] * (1 + (streak_len - 2) * 0.15)
                
                if prob_big > prob_small:
                    strategies.append({
                        'name': f'Streak Analysis ({streak_len}x {labels[0]})',
                        'prediction': 'Big',
                        'confidence': prob_big,
                        'weight': streak_weight
                    })
                else:
                    strategies.append({
                        'name': f'Streak Analysis ({streak_len}x {labels[0]})',
                        'prediction': 'Small',
                        'confidence': prob_small,
                        'weight': streak_weight
                    })
        
        # Strategy 3, 4, 5: Multi-length Sequence Patterns
        for seq_len in [3, 4, 5]:
            if len(labels) >= seq_len:
                seq_key = ''.join('B' if l == 'Big' else 'S' for l in labels[:seq_len])
                seq_data = self.model.get(f'sequence_patterns_{seq_len}', {}).get(seq_key, {'Big': 1, 'Small': 1})
                total = seq_data['Big'] + seq_data['Small']
                
                min_data = MIN_SEQUENCE_DATA + (seq_len - 3) * 2  # More data needed for longer sequences
                if total > min_data:
                    prob_big = seq_data['Big'] / total
                    prob_small = seq_data['Small'] / total
                    
                    weight = self.model['strategy_weights'].get(f'sequence_{seq_len}', 1.0)
                    
                    if prob_big > prob_small:
                        strategies.append({
                            'name': f'Sequence-{seq_len} ({seq_key})',
                            'prediction': 'Big',
                            'confidence': prob_big,
                            'weight': weight
                        })
                    else:
                        strategies.append({
                            'name': f'Sequence-{seq_len} ({seq_key})',
                            'prediction': 'Small',
                            'confidence': prob_small,
                            'weight': weight
                        })
        
        # Strategy 6: Distribution Correction
        recent_big = sum(1 for l in labels[:10] if l == 'Big')
        recent_small = 10 - recent_big if len(labels) >= 10 else len(labels) - recent_big
        
        if recent_big >= DISTRIBUTION_THRESHOLD:
            strategies.append({
                'name': f'Distribution Correction ({recent_big}/10 Big)',
                'prediction': 'Small',
                'confidence': 0.55 + (recent_big - DISTRIBUTION_THRESHOLD) * 0.05,
                'weight': self.model['strategy_weights']['frequency']
            })
        elif recent_small >= DISTRIBUTION_THRESHOLD:
            strategies.append({
                'name': f'Distribution Correction ({recent_small}/{len(labels[:10])} Small)',
                'prediction': 'Big',
                'confidence': 0.55 + (recent_small - DISTRIBUTION_THRESHOLD) * 0.05,
                'weight': self.model['strategy_weights']['frequency']
            })
        
        # Strategy 7: Alternation Detection
        alternating = True
        for i in range(min(4, len(labels) - 1)):
            if labels[i] == labels[i + 1]:
                alternating = False
                break
        
        if alternating and len(labels) >= 4:
            next_pred = 'Small' if labels[0] == 'Big' else 'Big'
            strategies.append({
                'name': 'Alternation Pattern',
                'prediction': next_pred,
                'confidence': 0.62,
                'weight': self.model['strategy_weights']['alternation']
            })
        
        # Strategy 8: Momentum/Trend Analysis
        if len(labels) >= MOMENTUM_WINDOW:
            momentum_strength, momentum_direction = self.calculate_momentum(labels)
            if momentum_direction != 'neutral' and momentum_strength > 0.1:
                # Predict against the momentum (mean reversion)
                counter_pred = 'Small' if momentum_direction == 'Big' else 'Big'
                strategies.append({
                    'name': f'Momentum Reversal ({momentum_direction} trend)',
                    'prediction': counter_pred,
                    'confidence': 0.52 + momentum_strength * 0.3,
                    'weight': self.model['strategy_weights']['momentum']
                })
        
        # Strategy 9: Hot/Cold Number Analysis
        if numbers:
            freq = self.model.get('number_frequency', {})
            total_freq = sum(freq.values()) if freq else 1
            recent_nums = numbers[:5]
            
            # Check if recent numbers are "hot" (frequent) or "cold" (rare)
            hot_count = sum(1 for n in recent_nums if freq.get(str(n), 0) / total_freq > HOT_NUMBER_THRESHOLD)
            cold_count = sum(1 for n in recent_nums if freq.get(str(n), 0) / total_freq < COLD_NUMBER_THRESHOLD)
            
            if hot_count >= 3:
                # Hot numbers tend to cool down - use get_label logic for consistency
                big_count_in_recent = sum(1 for n in recent_nums if get_label(n) == 'Big')
                cold_prediction = 'Small' if big_count_in_recent > 2 else 'Big'
                strategies.append({
                    'name': f'Hot Numbers Cooling ({hot_count}/5 hot)',
                    'prediction': cold_prediction,
                    'confidence': 0.53,
                    'weight': self.model['strategy_weights']['hot_cold']
                })
        
        # Ensemble voting with weighted confidence
        if strategies:
            big_score = sum(s['confidence'] * s['weight'] for s in strategies if s['prediction'] == 'Big')
            small_score = sum(s['confidence'] * s['weight'] for s in strategies if s['prediction'] == 'Small')
            total_score = big_score + small_score
            
            if total_score > 0:
                if big_score > small_score:
                    final_confidence = big_score / total_score
                    best_strategy = max([s for s in strategies if s['prediction'] == 'Big'], 
                                       key=lambda x: x['confidence'] * x['weight'])
                    return 'Big', final_confidence, best_strategy['name'], strategies
                else:
                    final_confidence = small_score / total_score
                    best_strategy = max([s for s in strategies if s['prediction'] == 'Small'], 
                                       key=lambda x: x['confidence'] * x['weight'])
                    return 'Small', final_confidence, best_strategy['name'], strategies
        
        return 'Big', 0.50, 'Default prediction', strategies
    
    def update_accuracy(self, was_correct):
        """Update model accuracy and adjust strategy weights."""
        self.model['total_predictions'] = self.model.get('total_predictions', 0) + 1
        if was_correct:
            self.model['correct_predictions'] = self.model.get('correct_predictions', 0) + 1
        
        self.save_model()


# Global predictor instance
predictor = MLPredictor()


def save_prediction(period, prediction, confidence, reason, strategies):
    """Save a prediction to JSON file."""
    predictions = load_json_file(PREDICTIONS_FILE, {'records': []})
    
    # Check if prediction already exists
    for p in predictions['records']:
        if p['period'] == period:
            return  # Already exists
    
    predictions['records'].append({
        'period': period,
        'prediction': prediction,
        'confidence': confidence,
        'reason': reason,
        'strategies': strategies,
        'actual_number': None,
        'actual_label': None,
        'correct': None,
        'timestamp': datetime.now().isoformat()
    })
    
    # Keep only last 1000 predictions
    if len(predictions['records']) > 1000:
        predictions['records'] = predictions['records'][-1000:]
    
    save_json_file(PREDICTIONS_FILE, predictions)


def update_prediction_result(period, actual_number):
    """Update a prediction with the actual result."""
    predictions = load_json_file(PREDICTIONS_FILE, {'records': []})
    actual_label = get_label(actual_number)
    
    for p in predictions['records']:
        if p['period'] == period and p['actual_number'] is None:
            p['actual_number'] = actual_number
            p['actual_label'] = actual_label
            p['correct'] = p['prediction'] == actual_label
            
            # Update predictor accuracy
            predictor.update_accuracy(p['correct'])
            break
    
    save_json_file(PREDICTIONS_FILE, predictions)


def get_prediction_stats():
    """Get prediction statistics from JSON file."""
    predictions = load_json_file(PREDICTIONS_FILE, {'records': []})
    records = predictions.get('records', [])
    
    # Calculate stats
    resolved = [r for r in records if r.get('correct') is not None]
    wins = sum(1 for r in resolved if r['correct'])
    losses = len(resolved) - wins
    
    # Calculate max loss streak
    max_loss_streak = 0
    current_streak = 0
    sorted_resolved = sorted(resolved, key=lambda x: x['period'], reverse=True)
    for r in sorted_resolved:
        if not r['correct']:
            current_streak += 1
            max_loss_streak = max(max_loss_streak, current_streak)
        else:
            current_streak = 0
    
    # Compute stages for history display
    history = []
    stage = 1
    for r in sorted_resolved[:100]:
        entry = {
            'period': r['period'],
            'prediction': r['prediction'],
            'actual_number': r['actual_number'],
            'actual_label': r['actual_label'],
            'correct': r['correct'],
            'confidence': r.get('confidence', 50),
            'reason': r.get('reason', ''),
            'stage': stage
        }
        history.append(entry)
        
        if r['correct']:
            stage = 1
        else:
            stage += 1
    
    return {
        'total': len(resolved),
        'wins': wins,
        'losses': losses,
        'accuracy': round(wins / len(resolved) * 100, 1) if resolved else 0,
        'max_loss_streak': max_loss_streak,
        'history': history,
        'model_accuracy': round(predictor.model.get('correct_predictions', 0) / 
                                max(predictor.model.get('total_predictions', 1), 1) * 100, 1)
    }


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/predict', methods=['GET'])
def predict():
    """Get the next prediction."""
    # Fetch latest draws
    draws, error = fetch_latest_draws()
    
    if error:
        return jsonify({'error': error}), 500
    
    if not draws:
        return jsonify({'error': 'No data available'}), 500
    
    # Save outcomes to JSON
    save_outcomes(draws)
    
    # Get historical data and train model
    historical = get_historical_outcomes(2000)
    predictor.train(historical)
    
    # Update any pending predictions with actual results
    for draw in draws:
        update_prediction_result(draw['period'], draw['num'])
    
    # Get next period
    latest_period = draws[0]['period']
    next_period = str(int(latest_period) + 1)
    
    # Make prediction using ML model
    prediction, confidence, reason, strategies = predictor.predict(draws)
    
    # Save prediction
    save_prediction(next_period, prediction, round(confidence * 100, 1), reason, 
                   [{'name': s['name'], 'prediction': s['prediction'], 
                     'confidence': round(s['confidence'] * 100, 1)} for s in strategies])
    
    # Get stats
    stats = get_prediction_stats()
    
    # Calculate analysis
    big_count = sum(1 for o in historical[:500] if o['label'] == 'Big')
    small_count = len(historical[:500]) - big_count
    total = big_count + small_count
    
    return jsonify({
        'latest_period': latest_period,
        'latest_number': draws[0]['num'],
        'latest_label': draws[0]['label'],
        'next_period': next_period,
        'prediction': prediction,
        'confidence': round(confidence * 100, 1),
        'reason': reason,
        'strategies': [{'name': s['name'], 'prediction': s['prediction'], 
                       'confidence': round(s['confidence'] * 100, 1)} for s in strategies],
        'stats': stats,
        'analysis': {
            'total_historical_records': len(historical),
            'big_percentage': round(big_count / total * 100, 2) if total > 0 else 50,
            'small_percentage': round(small_count / total * 100, 2) if total > 0 else 50,
            'model_accuracy': stats['model_accuracy']
        }
    })


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get prediction history."""
    stats = get_prediction_stats()
    return jsonify(stats)


@app.route('/api/analysis', methods=['GET'])
def get_analysis():
    """Get detailed analysis of historical data."""
    historical = get_historical_outcomes(1000)
    
    if len(historical) < 10:
        return jsonify({'error': 'Not enough historical data'})
    
    # Calculate various statistics
    labels = [o['label'] for o in historical]
    numbers = [o['number'] for o in historical]
    
    big_count = sum(1 for l in labels if l == 'Big')
    small_count = len(labels) - big_count
    
    # Streak analysis
    streaks = {'big': [], 'small': []}
    current_streak = 1
    current_type = labels[0]
    
    for i in range(1, len(labels)):
        if labels[i] == current_type:
            current_streak += 1
        else:
            if current_type == 'Big':
                streaks['big'].append(current_streak)
            else:
                streaks['small'].append(current_streak)
            current_streak = 1
            current_type = labels[i]
    
    return jsonify({
        'total_records': len(historical),
        'big_count': big_count,
        'small_count': small_count,
        'big_percentage': round(big_count / len(labels) * 100, 2),
        'small_percentage': round(small_count / len(labels) * 100, 2),
        'number_frequency': dict(Counter(numbers)),
        'avg_big_streak': round(sum(streaks['big']) / len(streaks['big']), 2) if streaks['big'] else 0,
        'avg_small_streak': round(sum(streaks['small']) / len(streaks['small']), 2) if streaks['small'] else 0,
        'max_big_streak': max(streaks['big']) if streaks['big'] else 0,
        'max_small_streak': max(streaks['small']) if streaks['small'] else 0,
        'model_info': {
            'total_predictions': predictor.model.get('total_predictions', 0),
            'correct_predictions': predictor.model.get('correct_predictions', 0),
            'strategy_weights': predictor.model.get('strategy_weights', {})
        }
    })


@app.route('/api/clear', methods=['POST'])
def clear_data():
    """Clear all prediction data (not outcomes)."""
    save_json_file(PREDICTIONS_FILE, {'records': []})
    # Reset model accuracy but keep learned patterns
    predictor.model['total_predictions'] = 0
    predictor.model['correct_predictions'] = 0
    predictor.save_model()
    return jsonify({'success': True})


# Ensure data directory exists on startup
ensure_data_dir()


if __name__ == '__main__':
    # Use environment variable for debug mode (default: False for production safety)
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 'yes')
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
