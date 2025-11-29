"""
WinGo 30S Predictor - Python Backend
A Flask application that predicts WinGo outcomes using historical data analysis.
"""

import sqlite3
import os
import time
import logging
import requests
from flask import Flask, jsonify, render_template, request
from collections import Counter
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for prediction results
RESULT_WIN = 1
RESULT_LOSS = 0

app = Flask(__name__)

DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wingo_data.db')
API_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"


def get_db_connection():
    """Create a database connection."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database with required tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Table for storing historical outcomes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            period TEXT UNIQUE NOT NULL,
            number INTEGER NOT NULL,
            label TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table for storing predictions and results
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            period TEXT UNIQUE NOT NULL,
            prediction TEXT NOT NULL,
            actual_number INTEGER,
            actual_label TEXT,
            correct INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Index for faster queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_outcomes_period ON outcomes(period)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_period ON predictions(period)')
    
    conn.commit()
    conn.close()


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
        for item in data['data']['list'][:50]:  # Get more data for analysis
            draws.append({
                'period': item['issueNumber'],
                'num': int(item['number'])
            })
        return draws, None
    except requests.exceptions.RequestException as e:
        return None, str(e)
    except (KeyError, ValueError) as e:
        return None, str(e)


def save_outcomes(draws):
    """Save fetched outcomes to database."""
    if not draws:
        return
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    for draw in draws:
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO outcomes (period, number, label)
                VALUES (?, ?, ?)
            ''', (draw['period'], draw['num'], get_label(draw['num'])))
        except sqlite3.Error as e:
            # Log but continue - duplicate inserts are expected and handled by IGNORE
            logger.debug(f"Database insert ignored for period {draw['period']}: {e}")
    
    conn.commit()
    conn.close()


def get_historical_outcomes(limit=1000):
    """Get historical outcomes from database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT period, number, label FROM outcomes
        ORDER BY period DESC LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def analyze_patterns(outcomes):
    """Analyze patterns in historical data."""
    if len(outcomes) < 10:
        return {}
    
    analysis = {
        'total_records': len(outcomes),
        'big_count': sum(1 for o in outcomes if o['label'] == 'Big'),
        'small_count': sum(1 for o in outcomes if o['label'] == 'Small'),
        'big_percentage': 0,
        'small_percentage': 0,
        'streaks': {'big': [], 'small': []},
        'pattern_after_streak': {},
        'number_frequency': Counter(o['number'] for o in outcomes)
    }
    
    analysis['big_percentage'] = round(analysis['big_count'] / len(outcomes) * 100, 2)
    analysis['small_percentage'] = round(analysis['small_count'] / len(outcomes) * 100, 2)
    
    # Analyze streaks
    current_streak = 1
    current_type = outcomes[0]['label'] if outcomes else None
    
    for i in range(1, len(outcomes)):
        if outcomes[i]['label'] == current_type:
            current_streak += 1
        else:
            if current_type == 'Big':
                analysis['streaks']['big'].append(current_streak)
            else:
                analysis['streaks']['small'].append(current_streak)
            current_streak = 1
            current_type = outcomes[i]['label']
    
    # Calculate average streak lengths
    if analysis['streaks']['big']:
        analysis['avg_big_streak'] = round(sum(analysis['streaks']['big']) / len(analysis['streaks']['big']), 2)
    if analysis['streaks']['small']:
        analysis['avg_small_streak'] = round(sum(analysis['streaks']['small']) / len(analysis['streaks']['small']), 2)
    
    # Analyze what comes after streaks of different lengths
    for streak_len in range(2, 6):
        analysis['pattern_after_streak'][streak_len] = {'big_after_big': 0, 'small_after_big': 0, 
                                                        'big_after_small': 0, 'small_after_small': 0}
        
        for i in range(streak_len, len(outcomes)):
            # Check if we have a streak ending at position i-1
            is_streak = True
            streak_type = outcomes[i-1]['label']
            for j in range(1, streak_len):
                if outcomes[i-j]['label'] != streak_type:
                    is_streak = False
                    break
            
            if is_streak:
                next_label = outcomes[i]['label']
                if streak_type == 'Big':
                    if next_label == 'Big':
                        analysis['pattern_after_streak'][streak_len]['big_after_big'] += 1
                    else:
                        analysis['pattern_after_streak'][streak_len]['small_after_big'] += 1
                else:
                    if next_label == 'Big':
                        analysis['pattern_after_streak'][streak_len]['big_after_small'] += 1
                    else:
                        analysis['pattern_after_streak'][streak_len]['small_after_small'] += 1
    
    return analysis


def predict_next(recent_draws, analysis=None):
    """
    Enhanced prediction algorithm using historical data.
    Uses multiple strategies and picks the most confident one.
    """
    if not recent_draws or len(recent_draws) < 3:
        return 'Big', 0.5, 'Insufficient data'
    
    strategies = []
    
    # Strategy 1: Streak reversal (original algorithm)
    labels = [get_label(d['num']) for d in recent_draws[:5]]
    streak_count = 1
    for i in range(1, len(labels)):
        if labels[i] == labels[0]:
            streak_count += 1
        else:
            break
    
    if streak_count >= 3:
        opposite = 'Small' if labels[0] == 'Big' else 'Big'
        confidence = min(0.55 + (streak_count - 3) * 0.05, 0.75)
        strategies.append((opposite, confidence, f'Streak reversal after {streak_count} {labels[0]}s'))
    
    # Strategy 2: Pattern analysis from historical data
    if analysis and analysis.get('pattern_after_streak'):
        for streak_len in range(min(streak_count, 5), 1, -1):
            if streak_len in analysis['pattern_after_streak']:
                pattern_data = analysis['pattern_after_streak'][streak_len]
                
                if labels[0] == 'Big':
                    total = pattern_data['big_after_big'] + pattern_data['small_after_big']
                    if total > 10:
                        prob_small = pattern_data['small_after_big'] / total
                        if prob_small > 0.55:
                            strategies.append(('Small', prob_small, f'Historical pattern: {prob_small*100:.1f}% Small after {streak_len} Bigs'))
                        elif prob_small < 0.45:
                            strategies.append(('Big', 1 - prob_small, f'Historical pattern: {(1-prob_small)*100:.1f}% Big continues'))
                else:
                    total = pattern_data['big_after_small'] + pattern_data['small_after_small']
                    if total > 10:
                        prob_big = pattern_data['big_after_small'] / total
                        if prob_big > 0.55:
                            strategies.append(('Big', prob_big, f'Historical pattern: {prob_big*100:.1f}% Big after {streak_len} Smalls'))
                        elif prob_big < 0.45:
                            strategies.append(('Small', 1 - prob_big, f'Historical pattern: {(1-prob_big)*100:.1f}% Small continues'))
                break
    
    # Strategy 3: Overall distribution bias
    if analysis:
        big_pct = analysis.get('big_percentage', 50)
        small_pct = analysis.get('small_percentage', 50)
        
        # Count recent distribution
        recent_big = sum(1 for l in labels[:10] if l == 'Big')
        recent_small = len(labels[:10]) - recent_big
        
        # If recent distribution differs significantly from historical
        if recent_big >= 7 and big_pct < 55:
            strategies.append(('Small', 0.58, f'Recent bias correction: Too many Bigs ({recent_big}/10)'))
        elif recent_small >= 7 and small_pct < 55:
            strategies.append(('Big', 0.58, f'Recent bias correction: Too many Smalls ({recent_small}/10)'))
    
    # Strategy 4: Alternation detection
    alternating = True
    for i in range(min(4, len(labels) - 1)):
        if labels[i] == labels[i + 1]:
            alternating = False
            break
    
    if alternating and len(labels) >= 4:
        next_pred = 'Small' if labels[0] == 'Big' else 'Big'
        strategies.append((next_pred, 0.60, 'Alternation pattern detected'))
    
    # Choose best strategy
    if strategies:
        best = max(strategies, key=lambda x: x[1])
        return best
    
    # Default: slight bias towards less common outcome
    if analysis:
        if analysis.get('big_percentage', 50) > 52:
            return 'Small', 0.51, 'Default: Slight Small bias'
        elif analysis.get('small_percentage', 50) > 52:
            return 'Big', 0.51, 'Default: Slight Big bias'
    
    return 'Big', 0.50, 'Default prediction'


def save_prediction(period, prediction):
    """Save a prediction to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO predictions (period, prediction)
            VALUES (?, ?)
        ''', (period, prediction))
        conn.commit()
    except sqlite3.Error as e:
        # Log but continue - the prediction can still work without being saved
        logger.warning(f"Failed to save prediction for period {period}: {e}")
    finally:
        conn.close()


def update_prediction_result(period, actual_number):
    """Update a prediction with the actual result."""
    conn = get_db_connection()
    cursor = conn.cursor()
    actual_label = get_label(actual_number)
    
    cursor.execute('''
        UPDATE predictions 
        SET actual_number = ?, actual_label = ?, correct = (prediction = ?)
        WHERE period = ?
    ''', (actual_number, actual_label, actual_label, period))
    conn.commit()
    conn.close()


def get_prediction_stats():
    """Get prediction statistics."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN correct = 0 THEN 1 ELSE 0 END) as losses
        FROM predictions 
        WHERE correct IS NOT NULL
    ''')
    row = cursor.fetchone()
    
    # Get recent predictions for history
    cursor.execute('''
        SELECT period, prediction, actual_number, actual_label, correct
        FROM predictions
        ORDER BY period DESC
        LIMIT 100
    ''')
    history = [dict(r) for r in cursor.fetchall()]
    
    # Calculate max loss streak
    cursor.execute('''
        SELECT correct FROM predictions 
        WHERE correct IS NOT NULL 
        ORDER BY period DESC
    ''')
    results = [r['correct'] for r in cursor.fetchall()]
    
    max_loss_streak = 0
    current_streak = 0
    for result in results:
        if result == RESULT_LOSS:
            current_streak += 1
            max_loss_streak = max(max_loss_streak, current_streak)
        else:
            current_streak = 0
    
    conn.close()
    
    total = row['total'] or 0
    wins = row['wins'] or 0
    losses = row['losses'] or 0
    
    return {
        'total': total,
        'wins': wins,
        'losses': losses,
        'accuracy': round(wins / total * 100, 1) if total > 0 else 0,
        'max_loss_streak': max_loss_streak,
        'history': history
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
    
    # Save outcomes to database
    save_outcomes(draws)
    
    # Get historical data for analysis
    historical = get_historical_outcomes(500)
    analysis = analyze_patterns(historical)
    
    # Update any pending predictions with actual results
    for draw in draws:
        update_prediction_result(draw['period'], draw['num'])
    
    # Get next period
    latest_period = draws[0]['period']
    next_period = str(int(latest_period) + 1)
    
    # Make prediction
    prediction, confidence, reason = predict_next(draws, analysis)
    
    # Save prediction
    save_prediction(next_period, prediction)
    
    # Get stats
    stats = get_prediction_stats()
    
    return jsonify({
        'latest_period': latest_period,
        'latest_number': draws[0]['num'],
        'latest_label': get_label(draws[0]['num']),
        'next_period': next_period,
        'prediction': prediction,
        'confidence': round(confidence * 100, 1),
        'reason': reason,
        'stats': stats,
        'analysis': {
            'total_historical_records': analysis.get('total_records', 0),
            'big_percentage': analysis.get('big_percentage', 50),
            'small_percentage': analysis.get('small_percentage', 50)
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
    analysis = analyze_patterns(historical)
    return jsonify(analysis)


@app.route('/api/clear', methods=['POST'])
def clear_data():
    """Clear all prediction data (not outcomes)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM predictions')
    conn.commit()
    conn.close()
    return jsonify({'success': True})


# Initialize database on startup
init_db()


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
