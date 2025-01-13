import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import linregress

class SmoothMovingAverage(object):
    """
    A class to compute the simple moving average (SMA) of a given symbol in a pandas DataFrame.

    Parameters:
    data (pd.DataFrame): Input data containing price information.
    symbol (str): The column name representing the symbol to calculate SMA for.
    window (int): The window size for calculating the moving average.
    """

    def __init__(self, data: pd.DataFrame, symbol: str, window: int):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        if symbol not in data.columns:
            raise ValueError(f"Symbol '{symbol}' not found in DataFrame columns.")
        if not isinstance(window, int) or window <= 0:
            raise ValueError("Window size must be a positive integer.")

        self.data = data
        self.symbol = symbol
        self.window = window
        self.sma = data[symbol].rolling(window=window).mean()

    def get_sma(self) -> pd.Series:
        """Returns the computed Simple Moving Average (SMA) as a pandas Series."""
        return self.sma

    def update_sma(self, new_data: pd.DataFrame):
        """Updates the SMA with new data."""
        if not isinstance(new_data, pd.DataFrame):
            raise ValueError("New data must be a pandas DataFrame.")
        if self.symbol not in new_data.columns:
            raise ValueError(f"Symbol '{self.symbol}' not found in new data columns.")
        self.sma = new_data[self.symbol].rolling(window=self.window).mean()


class ExponentialMovingAverage(object):
    """
    A class to compute the Exponential Moving Average (EMA) of a given symbol in a pandas DataFrame.

    Parameters:
    data (pd.DataFrame): Input data containing price information.
    symbol (str): The column name representing the symbol to calculate EMA for.
    window (int): The window size (span) for calculating the EMA.
    """

    def __init__(self, data: pd.DataFrame, symbol: str, window: int):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        if symbol not in data.columns:
            raise ValueError(f"Symbol '{symbol}' not found in DataFrame columns.")
        if not isinstance(window, int) or window <= 0:
            raise ValueError("Window size must be a positive integer.")

        self.data = data
        self.symbol = symbol
        self.window = window
        self.ema = data[symbol].ewm(span=window, adjust=True, ignore_na=True).mean()

    def get_ema(self) -> pd.Series:
        """Returns the computed Exponential Moving Average (EMA) as a pandas Series."""
        return self.ema

    def update_ema(self, new_data: pd.DataFrame):
        """Updates the EMA with new data."""
        if not isinstance(new_data, pd.DataFrame):
            raise ValueError("New data must be a pandas DataFrame.")
        if self.symbol not in new_data.columns:
            raise ValueError(f"Symbol '{self.symbol}' not found in new data columns.")
        self.ema = new_data[self.symbol].ewm(span=self.window, adjust=True, ignore_na=True).mean()

    

class RSI(object):
    def __init__(self, data, symbol, short_period=14, long_period=20):
     
        # 計算每日變動百分比
        self.diff_pct = data[symbol].diff()

        # 計算長期 RSI
        self.long_average_gain = self.diff_pct.where(self.diff_pct > 0, 0).ewm(span=long_period, adjust=False).mean()
        self.long_average_loss = -self.diff_pct.where(self.diff_pct < 0, 0).ewm(span=long_period, adjust=False).mean()
        
        self.longRS = self.long_average_gain / self.long_average_loss
        self.longRS = self.longRS.replace([np.inf, -np.inf], 0).fillna(0)
        self.longRSI = 100 - (100 / (1 + self.longRS))

        # 計算短期 RSI
        self.short_average_gain = self.diff_pct.where(self.diff_pct > 0, 0).ewm(span=short_period, adjust=False).mean()
        self.short_average_loss = -self.diff_pct.where(self.diff_pct < 0, 0).ewm(span=short_period, adjust=False).mean()
        self.shortRS = self.short_average_gain / self.short_average_loss
        self.shortRS = self.shortRS.replace([np.inf, -np.inf], 0).fillna(0)
        self.shortRSI = 100 - (100 / (1 + self.shortRS))
    
    def get_long_rsi(self):
        return self.longRSI

    def get_short_rsi(self):
        return self.shortRSI
'''
Reference:https://academy.binance.com/zt/articles/stochastic-rsi-explained
'''
class StochasticRSI(object):

    def __init__(self, data, symbol, period=20):
        self.rsi = RSI(data, symbol, period).get_short_rsi()
        self.period = period
        self.stochRSI = self.compute_stochastic_rsi()

    def compute_stochastic_rsi(self):
        # Calculate the minimum and maximum RSI values over the rolling period
        lowest_rsi = self.rsi.rolling(window=self.period, min_periods=1).min()
        highest_rsi = self.rsi.rolling(window=self.period, min_periods=1).max()
        # Calculate the Stochastic RSI
        stoch_rsi = (self.rsi - lowest_rsi) / (highest_rsi - lowest_rsi)
        
        # Handle division by zero and fill NaN values appropriately
        stoch_rsi = stoch_rsi.replace([np.inf, -np.inf], np.nan).fillna(0)
        return stoch_rsi

    def get_stochastic_rsi(self):
        
        return self.stochRSI

class OBV(object):

    def __init__(self,data) :
        self.close = data["Close"]
        self.volume = data["Volume"]
        self.obv = self.compute_OBV()
        pass
    def compute_OBV(self):
        close_diff = self.close.diff()
        direction = close_diff.apply(lambda x: 1 if x > 0 else (-1 if x<0 else 0))
        volume_adjust = self.volume*direction
        obv = volume_adjust.cumsum()
        obv.iloc[0]=0
       
        return obv
    def get_OBV(self):
        return self.obv

class BollingerBands(object):
    def __init__(self,data):
        self.close = data["Close"]
        self.sma20 = SmoothMovingAverage(data,"Close",20).get_sma()
        self.std = data["Close"].rolling(window=20).std()
        self.upper_band = self.sma20+2*self.std
        self.lower_band = self.sma20-2*self.std
        print(f"sma:\n{self.sma20.head()}")
        print(f"upper:\n{self.upper_band.head()}")
        print(f"upper:\n{self.lower_band.head()}")
    def get_plot(self):
        plt.figure(figsize=(12,6))
        plt.plot(self.close.index, self.close, label='Close', color='black')
        plt.plot(self.sma20.index, self.sma20, label='20 SMA', color='blue')
        plt.plot(self.upper_band.index, self.upper_band, label='upper band', color='green')
        plt.plot(self.lower_band.index, self.lower_band, label='lower band', color='red')
        plt.fill_between(self.upper_band.index, self.upper_band, self.lower_band, color='grey', alpha=0.1)
        plt.title('Bollinger Band')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(loc='best')
        plt.show()

    def get_upper_band(self):
        return self.upper_band

    def get_lower_band(self):
        return self.lower_band

    def get_middle_line(self):
        return self.sma20


class KeltnerChannel(object):
    def __init__(self,data,period = 14) -> None:
        self.close = data["Close"]
        self.high = data["High"]
        self.low = data["Low"]
        self.atr = self.compute_ATR(period)
        self.ema20=ExponentialMovingAverage(data,"Close",20).get_ema()
        self.upper_band = self.ema20+2*self.atr
        self.lower_band = self.ema20-2*self.atr

    def compute_ATR(self,period):
        prev_close = self.close.shift(periods = 1) # 移動資料
        #TR1: Today High - Low
        TR1=self.high-self.low
        #TR2: Today High -yesterday's close
        TR2 = abs(self.high-prev_close)
        #TR3: yesterday's close - today's low
        TR3 = abs(prev_close - self.low)
        # ATR : max(TR1,TR2,TR3)
        TR = pd.DataFrame({"TR1":TR1,"TR2":TR2,"TR3":TR3})
        TR = TR.max(axis=1)
        ATR = ExponentialMovingAverage(TR.to_frame(name ="tr"),'tr',period).get_ema()
       
        return ATR
    def get_plot(self):
        plt.figure(figsize=(12,6))
        plt.plot(self.close.index, self.close, label='close', color='black')
        plt.plot(self.ema20.index, self.ema20, label='20 EMA', color='blue')
        plt.plot(self.upper_band.index, self.upper_band, label='upper band', color='green')
        plt.plot(self.lower_band.index, self.lower_band, label='lower band', color='red')
        plt.fill_between(self.upper_band.index, self.upper_band, self.lower_band, color='grey', alpha=0.1)
        plt.title('Keltner Channel')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(loc='best')
        plt.show()
    def get_upper_band(self):
        return self.upper_band

    def get_lower_band(self):
        return self.lower_band

    def get_middle_line(self):
        return self.ema20


class MACD(object):
    """
    A class to calculate the MACD (Moving Average Convergence Divergence) indicator.

    Parameters:
    data (pd.DataFrame): Input data containing price information.
    symbol (str): The column name representing the price to calculate MACD for.
    short_period (int): The short EMA period (default: 5).
    long_period (int): The long EMA period (default: 35).
    sigal_period (int): The signal EMA period (default: 5).
    """

    def __init__(self, data: pd.DataFrame, symbol: str = 'Close', short_period: int = 5, long_period: int = 35, sigal_period: int = 5):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        if symbol not in data.columns:
            raise ValueError(f"Symbol '{symbol}' not found in DataFrame columns.")
        if not isinstance(short_period, int) or short_period <= 0:
            raise ValueError("Short period must be a positive integer.")
        if not isinstance(long_period, int) or long_period <= 0:
            raise ValueError("Long period must be a positive integer.")
        if not isinstance(sigal_period, int) or sigal_period <= 0:
            raise ValueError("Signal period must be a positive integer.")
        if short_period >= long_period:
            raise ValueError("Short period must be less than long period.")
        if len(data) < max(short_period, long_period, sigal_period):
            raise ValueError("Not enough data to calculate MACD.")

        self.data = data
        self.symbol = symbol
        self.short_period = short_period
        self.long_period = long_period
        self.sigal_period = sigal_period

        self.long_ema = ExponentialMovingAverage(data, symbol, long_period).get_ema()
        self.short_ema = ExponentialMovingAverage(data, symbol, short_period).get_ema()
        self.macd = self.short_ema - self.long_ema
        macd_df = pd.DataFrame({'MACD': self.macd})
        self.signal = ExponentialMovingAverage(macd_df, "MACD", sigal_period).get_ema()

    def get_macd_line(self) -> pd.Series:
        """Returns the MACD line as a pandas Series."""
        return self.macd

    def get_signal_line(self) -> pd.Series:
        """Returns the signal line as a pandas Series."""
        return self.signal

    def get_histogram_line(self) -> pd.Series:
        """Returns the MACD histogram (difference between MACD and signal line)."""
        return self.macd - self.signal

    def get_long_ema(self) -> pd.Series:
        """Returns the long EMA as a pandas Series."""
        return self.long_ema

    def get_short_ema(self) -> pd.Series:
        """Returns the short EMA as a pandas Series."""
        return self.short_ema

class AdaptiveEMA:
    def __init__(self, data: pd.DataFrame, base_period: int, weights:dict=None):
        '''
        Calculate adaptive EMA period based on market characteristics.

        Parameters:
        data: pandas DataFrame with columns 'Close', 'Volume', 'High', 'Low'
        base_period: integer starting EMA period to adjust
        weights: dict, coefficients of combined_factor with keys 'volatility', 'volume', 'range', 'momentum'
        '''
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        required_columns = {'Close', 'Volume', 'High', 'Low'}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

        self.data = data
        self.base_period = base_period
        self.weights = weights or {'volatility': 0.4, 'volume': 0.3, 'range': 0.2, 'momentum': 0.1}
        self.recommended_period, self.metrics = self._calculate_adaptive_ema_period()
        self.adaptive_ema = self.data['Close'].ewm(span=self.recommended_period).mean()

    def get_performance_metrics(self)->dict:
        return self.metrics
    def get_ema(self)->pd.Series:
        return self.adaptive_ema
    def calculate_adaptive_ema_period(self)->dict:
        """
        min_period: minimum allowed EMA period
        max_period: maximum allowed EMA period

        Returns:
        int: recommended EMA period
        dict: market metrics used in calculation
        """
        data = self.data.copy()
        base_period = self.base_period
        min_period = int(base_period/2)
        max_period = int(base_period*2)
        weights = self.weights
        TRADING_DAYS = 252
        # 1. 計算波動度（volatility）

        returns = data['Close'].pct_change()
        volatility = returns.std() * np.sqrt(TRADING_DAYS)  # 年化波動度

        # 2. 計算成交量資訊（volume profile）
        volume_mean = data['Volume'].mean()
        volume_cv = data['Volume'].std() / volume_mean  # 成交量的變異係數（Coefficient of Variation, CV）

        # 3. 計算價格區間
        avg_range = (data['High'] - data['Low']).mean() / data['Close'].mean()

        # 4. 計算動能（momentum）
        momentum = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1

        # 5. 根據不同的市場衡量值，計算對基準週期的調整因子
        #    (a) volatility_factor：波動度因子
        #        波動度越高，越傾向縮短週期（因為想要更即時地反應市場狀況）
        #        這裡以 (1 - (volatility / 2)) 作為衡量，高波動度時會降低此因子
        volatility_factor = 1 - (volatility / 2)

        #    (b) volume_factor：成交量因子
        #        交易量越穩定(變異係數低)，越傾向拉長週期（因為市場環境較為平穩）
        #        這裡以 (1 + volume_cv) 作為衡量，若成交量變異大，就會提高此因子
        volume_factor = 1 + (volume_cv)

        #    (c) range_factor：價格區間因子
        #        價格波動區間大，表示市場波動更激烈，週期宜縮短
        #        這裡以 (1 - (avg_range * 10)) 作為衡量，若 avg_range 大，則因子會被減小
        range_factor = 1 - (avg_range * 10)

        #    (d) momentum_factor：動能因子
        #        趨勢方向越強（正或負），越傾向拉長週期，追蹤長期趨勢
        #        這裡以 (1 + abs(momentum)) 作為衡量，動能大則因子增大
        momentum_factor = 1 + abs(momentum)

        # 6. 將各因子以不同權重加總，形成最終的綜合因子（combined_factor）
        #    此處權重為：波動度 0.4、成交量 0.3、價格區間 0.2、動能 0.1
        combined_factor = (
            volatility_factor * weights['volatility'] +
            volume_factor * weights['volume'] +
            range_factor * weights['range'] +
            momentum_factor * weights['momentum']
        )

        # 7. 以 base_period 為基準，乘上綜合因子，得到建議週期
        adjusted_period = int(base_period * combined_factor)

        # 8. 將建議週期限制在 min_period 和 max_period 之間
        final_period = max(min_period, min(adjusted_period, max_period))

        # 9. 將各項指標與最終結果打包到字典中回傳
        metrics = {
            'volatility': volatility,
            'volume_cv': volume_cv,
            'avg_range': avg_range,
            'momentum': momentum,
            'combined_factor': combined_factor,
            'recommended_period': final_period
        }

        return final_period, metrics

        
class TrendMarked():
    """
    A class to compute and visualize trends in time series data, including moving average, 
    short-term trends, peaks, valleys, and overall trend slopes.

    Parameters:
    data: pandas dataframe with columns "Datetime" and specific symbol
    symbol: str for choose which column to compute trends
    windows: distance for scipy.find_peaks. distance:Required minimal horizontal distance (>= 1) in samples between neighbouring peaks.
    ma_window: moving average window
    """

    def __init__(self, data: pd.DataFrame, symbol: str, window: int, ma_window: int = 5):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        required_columns = {'Datetime', symbol}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
        if not isinstance(window, int) or window <= 0:
            raise ValueError("Window size must be a positive integer.")
        if not isinstance(ma_window, int) or ma_window <= 0:
            raise ValueError("Moving average window must be a positive integer.")
        
        self.data = data
        self.symbol = symbol
        self.window = window
        self.ma_window = ma_window
        self.trends = self.compute()

    def compute(self) -> pd.DataFrame:
        df = self.data.copy()
        df['moving_avg'] = df[self.symbol].rolling(window=self.ma_window).mean()
        df['diff'] = df[self.symbol].diff()
        df['short_term_trend'] = df['diff'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        
        peaks, _ = find_peaks(df[self.symbol], distance=self.window)
        valleys, _ = find_peaks(-df[self.symbol], distance=self.window)
        
        df['is_peak'] = False
        df.loc[peaks, 'is_peak'] = True
        df['is_valley'] = False
        df.loc[valleys, 'is_valley'] = True
        
        df['slope'] = df[self.symbol].rolling(window=self.window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
        )
        df['trend'] = df['slope'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        
        return df

    def get_trends(self) -> pd.DataFrame:
        """Returns the DataFrame containing trend information."""
        return self.trends

    def get_plot(self, show_peaks_valleys=True, save_path=None):
        """Plots the original data, moving average, and optionally peaks/valleys."""
        df = self.trends
        plt.figure(figsize=(10, 6))
        plt.plot(df['Datetime'], df[self.symbol], label='Original Data', color='black')
        plt.plot(df['Datetime'], df['moving_avg'], label='Moving Average', linestyle='--', color='orange')

        if show_peaks_valleys:
            plt.scatter(df['Datetime'][df['is_peak']], df[self.symbol][df['is_peak']], color='red', label='Peaks', zorder=5)
            plt.scatter(df['Datetime'][df['is_valley']], df[self.symbol][df['is_valley']], color='blue', label='Valleys', zorder=5)

        plt.title('Time Series Trend, Peaks, and Valleys')
        plt.xlabel('Time')
        plt.ylabel(self.symbol)
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.show()


if __name__=='__main__':
    # 載入資料
    # data = pd.read_csv("btc_usd_20231013_20241113.csv")
    # print(data.head())
    # instance = StochasticRSI(data,"Close")
    # rsi = instance.get_stochastic_rsi()*100
    # ema = ExponentialMovingAverage(data,'Close',5).get_ema()
    # rsi.to_csv("rsi.csv")
    # ema.to_csv("ema.csv")
    # plt.figure(figsize=(16,9))
    # #plt.plot(data['Close'],color = 'blue',label ='close')
    # plt.plot(rsi,color = 'blue',label='rsi')
    # plt.plot([20]*len(rsi),color = 'gold',label = '70',lw = 2)
    # plt.plot([80]*len(rsi),color = 'gold',label = '70',lw = 2)
    # #plt.plot(ema,color = "red",label='ema5')
    # plt.show()
    df = pd.read_csv(r"C:\Users\louislin\OneDrive\桌面\data_analysis\backtesting_system\data\1d_BNBUSDT.csv",index_col='Datetime')
    print(df.iloc[0:60])
    indicator = trend(df.iloc[0:60],"Close",10)
    # print(indicator.get_trends())
    # print(indicator.get_plot())
    # macd = MACD(df.iloc[0:60],'Close',14,26,9)
    # diff = macd.get_MACD()
    # dem = macd.get_signal()
    # print(f"MACD INFO:\nDIFF:\n{diff}DEM:\n{dem}")
    # diff_trend = trend(diff,'Close',5)
    # print(diff_trend.get_trends())

    
  
    
    
    