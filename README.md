# 📈 Stock Market ML Prediction Model

A machine learning pipeline for predicting next-day stock price direction (UP/DOWN) using technical indicators. Built and tested on **Coal India (COALINDIA.NS)** historical data.

---

## 🚀 Features

- **Data ingestion & cleaning** — loads raw OHLCV CSV data, deduplicates, and handles missing values
- **Technical indicator engineering** — 14 indicators including SMA, EMA, MACD, RSI, Bollinger Bands, ATR, and Volume Ratio
- **Multi-model training** — trains and compares Logistic Regression, Random Forest, Gradient Boosting, and Neural Network classifiers
- **Automatic model selection** — picks the best model based on validation accuracy
- **Trading signal generation** — generates BUY / SELL / HOLD signals with a configurable confidence threshold
- **Backtesting engine** — simulates a simple long-only strategy on historical data
- **Model persistence** — saves the trained model, scaler, and feature list via `joblib`

---

## 🗂️ Project Structure

```
├── stock_prediction.py          # Main pipeline script
├── COALINDIA.csv                # Input data (OHLCV format)
├── coalindia_model.pkl          # Saved best model (generated)
├── coalindia_model_scaler.pkl   # Saved StandardScaler (generated)
├── coalindia_model_features.pkl # Saved feature list (generated)
├── feature_importance.png       # Feature importance chart (generated)
└── README.md
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

Install all dependencies with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

---

## 📊 Input Data Format

The CSV file must contain the following columns:

| Column   | Description               |
|----------|---------------------------|
| `Date`   | Trading date (parseable)  |
| `Open`   | Opening price             |
| `High`   | Daily high                |
| `Low`    | Daily low                 |
| `Close`  | Closing price             |
| `Volume` | Volume traded             |

---

## ⚙️ Usage

1. Place your CSV file in the project directory (or update the path inside `load_and_clean_data`).

2. Run the full pipeline:

```bash
python stock_prediction.py
```

3. The script will:
   - Load and clean the data
   - Engineer technical indicators
   - Train 4 models and print accuracy for each
   - Evaluate the best model in detail (accuracy, F1, ROC-AUC, confusion matrix)
   - Print recent BUY/SELL signals
   - Run a backtest starting from ₹1,00,000 initial capital
   - Save the trained model files

---

## 🧠 Models Compared

| Model                | Notes                                      |
|----------------------|--------------------------------------------|
| Logistic Regression  | Baseline linear classifier                 |
| Random Forest        | 100 estimators, max depth 20               |
| Gradient Boosting    | 100 estimators, learning rate 0.1          |
| Neural Network (MLP) | Layers: 64 → 32, early stopping enabled   |

The model with the highest **validation accuracy** is automatically selected as the best model.

---

## 📐 Feature Set

| Feature         | Description                        |
|-----------------|------------------------------------|
| SMA_20/50/200   | Simple Moving Averages             |
| EMA_12/26       | Exponential Moving Averages        |
| MACD            | MACD line and signal line          |
| MACD_Diff       | MACD histogram                     |
| RSI             | 14-period Relative Strength Index  |
| BB_Position     | Bollinger Band relative position   |
| ATR             | 14-period Average True Range       |
| Volume_Ratio    | Volume vs 20-day average           |
| ROC             | 10-period Rate of Change           |
| Momentum        | 10-period price momentum           |

---

## 📉 Backtesting Logic

- Starts with ₹1,00,000 (configurable via `initial_capital`)
- Executes **BUY** when signal is BUY and no position is held
- Executes **SELL** when signal is SELL and a position is held
- Closes any open position at the last available price
- Reports final capital, total return (%), and number of trades

> ⚠️ The backtest is simplified and does not account for transaction costs, slippage, or taxes.

---

## 🔧 Configuration

| Parameter             | Location                       | Default           |
|-----------------------|--------------------------------|-------------------|
| Train/Val/Test split  | `prepare_ml_data()`            | 70 / 15 / 15      |
| Signal confidence     | `generate_trading_signals()`   | 0.65              |
| Initial capital       | `backtest()`                   | ₹1,00,000         |
| Model output filename | `save_model()`                 | `coalindia_model` |

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is not financial advice. Past performance of a backtested model does not guarantee future results. Always do your own due diligence before making investment decisions.

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
