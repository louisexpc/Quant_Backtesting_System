import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
class SmoothMovingAverage(object):
   
    def __init__(self,data,symbol,window):
        self.sma = data[symbol].rolling(window=window).mean()

    def get_sma(self):
        return self.sma

class ExponentialMovingAverage(object):

    def __init__(self, data, symbol, window):
        self.ema = data[symbol].ewm(span = window,adjust=True,ignore_na=True).mean()
            
    def get_ema(self):
        return self.ema
    

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
    def __init__(self,data,symbol='Close',short_period=5,long_period=35,sigal_period = 5):
        self.long_ema = ExponentialMovingAverage(data,symbol,long_period).get_ema()
        self.short_ema = ExponentialMovingAverage(data,symbol,short_period).get_ema()
        self.macd = (self.short_ema - self.long_ema)
        
        self.signal = ExponentialMovingAverage(self.macd.to_frame(name ="MACD"),"MACD",sigal_period).get_ema()
    def get_MACD(self):
        return self.macd
    def get_signal(self):
        return self.signal
    def get_histogram(self):
        return self.macd-self.signal
    def get_long_ema(self):
        return self.long_ema
    def get_short_ema(self):
        return self.short_ema
if __name__=='__main__':
    # 載入資料
    data = pd.read_csv("btc_usd_20231013_20241113.csv")
    print(data.head())
    instance = StochasticRSI(data,"Close")
    rsi = instance.get_stochastic_rsi()*100
    ema = ExponentialMovingAverage(data,'Close',5).get_ema()
    rsi.to_csv("rsi.csv")
    ema.to_csv("ema.csv")
    plt.figure(figsize=(16,9))
    #plt.plot(data['Close'],color = 'blue',label ='close')
    plt.plot(rsi,color = 'blue',label='rsi')
    plt.plot([20]*len(rsi),color = 'gold',label = '70',lw = 2)
    plt.plot([80]*len(rsi),color = 'gold',label = '70',lw = 2)
    #plt.plot(ema,color = "red",label='ema5')
    plt.show()
    
  
    
    
    