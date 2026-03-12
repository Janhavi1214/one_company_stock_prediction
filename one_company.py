import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns


# =====================================================
# STEP 1: DATA LOADING & CLEANING
# =====================================================
def load_and_clean_data(csv_file):
    """Load and clean stock data"""
    df = pd.read_csv( r"C:\Users\Janhavi\OneDrive\Desktop\Stock dataset\COALINDIA.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.drop_duplicates(subset=['Date'])
    df = df.dropna()

    print(f"✓ Data loaded: {len(df)} records")
    print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    return df


# =====================================================
# STEP 2: FEATURE ENGINEERING
# =====================================================
def calculate_technical_indicators(df):
    """Calculate all technical indicators"""

    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()

    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']

    # RSI
    def calculate_rsi(close, window=14):
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['RSI'] = calculate_rsi(df['Close'])

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (
        df['BB_Upper'] - df['BB_Lower']
    )

    # ATR
    df['High_Low'] = df['High'] - df['Low']
    df['High_Close'] = abs(df['High'] - df['Close'].shift())
    df['Low_Close'] = abs(df['Low'] - df['Close'].shift())
    df['TR'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()

    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']

    # Momentum & ROC
    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    df['Momentum'] = df['Close'] - df['Close'].shift(10)

    # Target (Up/Down next day)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    df = df.dropna()

    print("✓ Technical indicators calculated")
    print("  Features: 18 indicators")
    return df


# =====================================================
# STEP 3: PREPARE DATA
# =====================================================
def prepare_ml_data(df):
    features = [
        'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26',
        'MACD', 'MACD_Signal', 'MACD_Diff', 'RSI', 'BB_Position',
        'ATR', 'Volume_Ratio', 'ROC', 'Momentum'
    ]

    X = df[features].values
    y = df['Target'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    train_size = int(len(X) * 0.70)
    val_size = int(len(X) * 0.15)

    X_train = X_scaled[:train_size]
    y_train = y[:train_size]

    X_val = X_scaled[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]

    X_test = X_scaled[train_size + val_size:]
    y_test = y[train_size + val_size:]

    print("✓ Data split completed")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test, X_scaled, scaler, features


# =====================================================
# STEP 4: TRAIN MULTIPLE MODELS
# =====================================================
def train_models(X_train, y_train, X_val, y_val, X_test, y_test):
    models = {}
    results = {}

    print("\n============================================================")
    print("TRAINING MODELS")
    print("============================================================")

    # Logistic Regression
    print("\n1. Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr
    results['Logistic Regression'] = {
        'train': lr.score(X_train, y_train),
        'val': lr.score(X_val, y_val),
        'test': lr.score(X_test, y_test)
    }

    # Random Forest
    print("2. Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    results['Random Forest'] = {
        'train': rf.score(X_train, y_train),
        'val': rf.score(X_val, y_val),
        'test': rf.score(X_test, y_test)
    }

    # Gradient Boosting
    print("3. Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb
    results['Gradient Boosting'] = {
        'train': gb.score(X_train, y_train),
        'val': gb.score(X_val, y_val),
        'test': gb.score(X_test, y_test)
    }

    # Neural Network
    print("4. Neural Network...")
    nn = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=1000,
        early_stopping=True,
        random_state=42
    )
    nn.fit(X_train, y_train)
    models['Neural Network'] = nn
    results['Neural Network'] = {
        'train': nn.score(X_train, y_train),
        'val': nn.score(X_val, y_val),
        'test': nn.score(X_test, y_test)
    }

    print("\n============================================================")
    print("MODEL PERFORMANCE")
    print("============================================================")

    for name, scores in results.items():
        print(f"\n{name}:")
        print(f"  Train Accuracy: {scores['train']:.4f}")
        print(f"  Val Accuracy:   {scores['val']:.4f}")
        print(f"  Test Accuracy:  {scores['test']:.4f}")

    best_model_name = max(results, key=lambda x: results[x]['val'])
    best_model = models[best_model_name]

    print(f"\n✓ Best Model: {best_model_name}")
    print(f"  Validation Accuracy: {results[best_model_name]['val']:.4f}")

    return best_model, best_model_name, models, results


# =====================================================
# STEP 5: EVALUATION
# =====================================================
def evaluate_model(model, X_test, y_test, model_name):
    print("\n============================================================")
    print(f"DETAILED EVALUATION - {model_name}")
    print("============================================================")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("\nConfusion Matrix:")
    print(f"  TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['DOWN', 'UP']))

    return y_pred, y_proba


# =====================================================
# STEP 6: FEATURE IMPORTANCE (Tree Models Only)
# =====================================================
def plot_feature_importance(model, features, model_name):
    if not hasattr(model, 'feature_importances_'):
        return

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title(f'Feature Importance - {model_name}')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.close()

    print("\n✓ Feature Importance:")
    print(importance_df.to_string(index=False))


# =====================================================
# STEP 7: TRADING SIGNAL GENERATION
# =====================================================
def generate_trading_signals(df, model, X_scaled, confidence_threshold=0.65):
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    df['Prediction'] = y_pred
    df['Probability'] = y_proba
    df['Signal'] = 'HOLD'

    buy_mask = (y_pred == 1) & (y_proba > confidence_threshold)
    sell_mask = (y_pred == 0) & (y_proba < (1 - confidence_threshold))

    df.loc[buy_mask, 'Signal'] = 'BUY'
    df.loc[sell_mask, 'Signal'] = 'SELL'

    signals_df = df[df['Signal'] != 'HOLD'][
        ['Date', 'Close', 'Signal', 'Probability', 'RSI', 'SMA_20', 'SMA_50']
    ].tail(20)

    print("\n============================================================")
    print("RECENT TRADING SIGNALS")
    print("============================================================")
    print(signals_df.to_string(index=False))

    print(f"\nTotal BUY signals: {buy_mask.sum()}")
    print(f"Total SELL signals: {sell_mask.sum()}")

    return df


# =====================================================
# STEP 8: BACKTESTING
# =====================================================
def backtest(df, initial_capital=100000):
    print("\n============================================================")
    print("BACKTESTING")
    print("============================================================")

    capital = initial_capital
    position = 0
    trades = []
    portfolio_values = [capital]

    for _, row in df.iterrows():
        price = row['Close']

        # BUY
        if row['Signal'] == 'BUY' and position == 0:
            position = capital / price
            trades.append({
                'Date': row['Date'], 'Type': 'BUY',
                'Price': price, 'Shares': position
            })
            capital = 0

        # SELL
        elif row['Signal'] == 'SELL' and position > 0:
            capital = position * price
            trades.append({
                'Date': row['Date'], 'Type': 'SELL',
                'Price': price, 'Capital': capital
            })
            position = 0

        portfolio_values.append(
            capital if position == 0 else position * price
        )

    # Close last position
    if position > 0:
        capital = position * df.iloc[-1]['Close']

    total_return = ((capital - initial_capital) / initial_capital) * 100

    print(f"\nInitial Capital: {initial_capital}")
    print(f"Final Capital:   {capital:.2f}")
    print(f"Total Return:    {total_return:.2f}%")
    print(f"Total Trades:    {len(trades)}")

    return capital, total_return, trades


# =====================================================
# STEP 9: SAVE MODEL
# =====================================================
def save_model(model, scaler, features, filename='stock_model'):
    joblib.dump(model, f'{filename}.pkl')
    joblib.dump(scaler, f'{filename}_scaler.pkl')
    joblib.dump(features, f'{filename}_features.pkl')
    print(f"\n✓ Model saved as '{filename}.pkl'")


# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == "__main__":

    print("\n============================================================")
    print("STOCK MARKET ML PREDICTION MODEL")
    print("============================================================")

    df = load_and_clean_data("COALINDIA.csv")
    df = calculate_technical_indicators(df)

    X_train, y_train, X_val, y_val, X_test, y_test, X_scaled, scaler, features = \
        prepare_ml_data(df)

    best_model, best_model_name, models, results = \
        train_models(X_train, y_train, X_val, y_val, X_test, y_test)

    evaluate_model(best_model, X_test, y_test, best_model_name)

    plot_feature_importance(best_model, features, best_model_name)

    df = generate_trading_signals(df, best_model, X_scaled, confidence_threshold=0.65)

    final_capital, total_return, trades = backtest(df)

    save_model(best_model, scaler, features, filename="coalindia_model")

    print("\n============================================================")
    print("✓ ANALYSIS COMPLETE")
    print("============================================================")
    print(f"\nBest Model: {best_model_name}")
    print(f"Test Accuracy: {results[best_model_name]['test']:.4f}")
    print(f"Backtest Return: {total_return:.2f}%")
