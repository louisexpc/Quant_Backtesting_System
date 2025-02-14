import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def trend_quantified(data: pd.DataFrame, period: int) -> pd.Series:
    """
    Quantifies the trend based on EMA, RSI, Bollinger Bands, and ADX.
    
    Returns:
        pd.Series: Trend classification
        1: Uptrend, 0: Sideways, -1: Downtrend
    """
    # Ensure data has the necessary columns
    required_columns = {'Close', 'High', 'Low'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    # Calculate short and long EMAs
    short_ema = data['Close'].ewm(span=15, adjust=False).mean()
    long_ema = data['Close'].ewm(span=30, adjust=False).mean()

    # Calculate RSI
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=data.index).rolling(window=14).mean()
    avg_loss = pd.Series(loss, index=data.index).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Calculate Bollinger Bands
    rolling_mean = data['Close'].rolling(window=period).mean()
    rolling_std = data['Close'].rolling(window=period).std()
    bollinger_upper = rolling_mean + 2 * rolling_std
    bollinger_lower = rolling_mean - 2 * rolling_std

    # Calculate ADX (Average Directional Index)
    high = data['High']
    low = data['Low']
    close = data['Close']
    tr = pd.Series(np.maximum.reduce([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ]), index=data.index)
    atr = tr.rolling(window=14).mean()

    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    plus_di = 100 * (plus_dm.rolling(window=14).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14).sum() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=14).mean()

    # Combine features into a DataFrame
    features = pd.DataFrame({
        'short_ema': short_ema,
        'long_ema': long_ema,
        'rsi': rsi,
        'bollinger_upper': bollinger_upper,
        'bollinger_lower': bollinger_lower,
        'adx': adx,
        'close': data['Close']
    })

    # Add trend factor calculation
    trend = pd.Series(0, index=data.index, dtype=int)
    trend[(features['short_ema'] > features['long_ema']) & (features['rsi'] > 50) &
          (features['close'] < features['bollinger_upper'])] = 1
    trend[(features['short_ema'] < features['long_ema']) & (features['rsi'] < 50) &
          (features['close'] > features['bollinger_lower'])] = -1

    # Fill missing values with 0 to match original data length
    trend = trend.fillna(0).astype(int)

    # Return the trend aligned with the original index
    return trend



def trend_quantified_with_ml(data: pd.DataFrame, period: int) -> pd.Series:
    """
    Quantifies the trend based on EMA, RSI, Bollinger Bands, and ADX.
    Includes optional machine learning integration for improved classification.
    
    Returns:
        pd.Series: Trend classification
        1: Uptrend, 0: Sideways, -1: Downtrend
    """
    # Ensure data has the necessary columns
    if 'Close' not in data.columns:
        raise ValueError("DataFrame must contain 'Close' column.")

    # Calculate short and long EMAs
    short_ema = data['Close'].ewm(span=15, adjust=False).mean()
    long_ema = data['Close'].ewm(span=30, adjust=False).mean()

    # Calculate RSI
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Calculate Bollinger Bands
    rolling_mean = data['Close'].rolling(window=period).mean()
    rolling_std = data['Close'].rolling(window=period).std()
    bollinger_upper = rolling_mean + 2 * rolling_std
    bollinger_lower = rolling_mean - 2 * rolling_std

    # Calculate ADX (Average Directional Index)
    high = data['High']
    low = data['Low']
    close = data['Close']
    tr = np.maximum.reduce([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ])
    atr = tr.rolling(window=14).mean()

    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    plus_di = 100 * (plus_dm.rolling(window=14).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14).sum() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=14).mean()

    # Combine features into a DataFrame
    features = pd.DataFrame({
        'short_ema': short_ema,
        'long_ema': long_ema,
        'rsi': rsi,
        'bollinger_upper': bollinger_upper,
        'bollinger_lower': bollinger_lower,
        'adx': adx,
        'close': data['Close']
    }).dropna()

    # Add trend factor calculation
    trend = pd.Series(0, index=features.index, dtype=int)
    trend[(features['short_ema'] > features['long_ema']) & (features['rsi'] > 50) &
          (features['close'] < features['bollinger_upper'])] = 1
    trend[(features['short_ema'] < features['long_ema']) & (features['rsi'] < 50) &
          (features['close'] > features['bollinger_lower'])] = -1

    # Optional: Train a Random Forest Classifier for enhanced prediction
    X = features[['short_ema', 'long_ema', 'rsi', 'adx']]
    y = trend
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict trend using the trained model
    predicted_trend = model.predict(X_scaled)

    # Return the predicted trend aligned with the original index
    return pd.Series(predicted_trend, index=features.index)
if __name__=='__main__':
    data = pd.read_csv(r"C:\Users\louislin\OneDrive\桌面\data_analysis\backtesting_system\data\15m_XRPUSDT.csv")
    print(data)
    result = trend_quantified(data,15)
    result.to_csv('test.csv')
    print(result.value_counts())
