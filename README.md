# WinGo 30S Predictor - Python Edition

A Python-powered prediction system for WinGo 30S game that uses historical data analysis to make predictions.

## Features

- **Persistent Data Storage**: All outcomes are saved in SQLite database, making predictions improve over time
- **Advanced Prediction Algorithm**: Uses multiple strategies including:
  - Streak reversal detection
  - Historical pattern analysis
  - Distribution bias correction
  - Alternation pattern detection
- **Real-time Updates**: Polls for new data and updates predictions automatically
- **Confidence Scoring**: Shows prediction confidence based on the strategy used
- **Strategy Transparency**: Displays the reasoning behind each prediction

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

4. Open your browser and navigate to `http://localhost:5000`

## Project Structure

```
wingo-predictor/
├── app.py              # Main Flask application with prediction logic
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Frontend UI
├── index.html          # Original HTML-only version (legacy)
└── README.md           # This file
```

## API Endpoints

- `GET /` - Main application UI
- `GET /api/predict` - Get the next prediction with confidence and reasoning
- `GET /api/history` - Get prediction history and statistics
- `GET /api/analysis` - Get detailed analysis of historical data
- `POST /api/clear` - Clear prediction history (keeps outcome data)

## How It Works

1. **Data Collection**: The system fetches lottery outcomes from the official API and stores them in a SQLite database.

2. **Pattern Analysis**: Analyzes historical data to find:
   - Big/Small distribution percentages
   - Streak patterns and their frequencies
   - What typically comes after streaks of different lengths

3. **Prediction Strategies**:
   - **Streak Reversal**: Predicts opposite after 3+ consecutive same outcomes
   - **Historical Patterns**: Uses past data to predict what comes after specific patterns
   - **Bias Correction**: Adjusts predictions if recent results deviate from historical norms
   - **Alternation Detection**: Recognizes alternating patterns

4. **Learning**: The more data collected, the better the pattern analysis becomes, improving prediction accuracy over time.

## Made By

Ali Abbas  
WhatsApp: +92 348 3469617
