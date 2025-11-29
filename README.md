# WinGo 30S Predictor - Python ML Edition

A Python-powered prediction system for WinGo 30S game that uses machine learning-style pattern analysis and historical data to make predictions. Data is stored in JSON files for persistence.

## Features

- **JSON-based Storage**: All outcomes, predictions, and model data stored in JSON files for easy portability
- **ML-Style Prediction Engine**: Uses multiple intelligent strategies:
  - **Markov Chain**: Transition probability analysis
  - **Streak Pattern Learning**: Learns from historical streak outcomes
  - **Sequence Pattern Recognition**: 3-tuple sequence-based predictions
  - **Distribution Bias Correction**: Adjusts for recent distribution anomalies
  - **Alternation Detection**: Recognizes alternating patterns
- **Ensemble Voting**: Combines multiple strategies with weighted confidence
- **Real-time Updates**: Polls for new data and updates predictions automatically
- **Confidence Scoring**: Shows prediction confidence with color-coded visualization
- **Strategy Transparency**: Displays all active strategies and their predictions
- **Win/Loss Tracking**: Tracks prediction accuracy with stages and streaks
- **Model Accuracy**: Shows overall model performance

## Requirements

- Python 3.8+
- Flask
- Requests

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/BeingAliAbbas/wingo-predictor.git
   cd wingo-predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

   For development with debug mode enabled:
   ```bash
   FLASK_DEBUG=true python app.py
   ```

4. Open your browser and navigate to `http://localhost:5000`

## Project Structure

```
wingo-predictor/
├── app.py              # Main Flask application with ML prediction logic
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Frontend UI
├── data/               # Auto-created directory for JSON storage
│   ├── outcomes.json   # Historical outcomes
│   ├── predictions.json # Prediction history
│   └── model.json      # Trained model data
├── index.html          # Original HTML-only version (legacy)
└── README.md           # This file
```

## API Endpoints

- `GET /` - Main application UI
- `GET /api/predict` - Get the next prediction with confidence, reasoning, and all strategies
- `GET /api/history` - Get prediction history and statistics
- `GET /api/analysis` - Get detailed analysis including model info
- `POST /api/clear` - Clear prediction history (keeps learned patterns)

## How It Works

### 1. Data Collection
The system fetches lottery outcomes from the official API and stores them in `data/outcomes.json`.

### 2. ML Model Training
The model learns from historical data:
- **Markov Chain**: Builds transition probabilities (Big→Big, Big→Small, etc.)
- **Streak Patterns**: Learns what typically follows streaks of 2-7 consecutive outcomes
- **Sequence Patterns**: Maps 3-outcome sequences to next outcome probabilities

### 3. Prediction Strategies
Multiple strategies run simultaneously:
- **Markov Chain**: Uses learned transition probabilities
- **Streak Analysis**: Considers current streak length and historical patterns
- **Sequence Pattern**: Matches current 3-outcome sequence to historical data
- **Distribution Correction**: Adjusts when recent results deviate significantly
- **Alternation Detection**: Recognizes ongoing alternating patterns

### 4. Ensemble Voting
Strategies vote on the prediction with their confidence levels. The final prediction is based on weighted voting where higher confidence strategies have more influence.

### 5. Continuous Learning
The model updates with each new outcome, improving accuracy over time.

## Made By

Ali Abbas  
WhatsApp: +92 348 3469617
