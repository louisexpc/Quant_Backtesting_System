import pandas as pd
import numpy as np
import os
import time
import talib

from pkg.ConfigLoader import config

from utils.trend_classification import trend_quantified

STRATEGY_CONFIG = "C:\\Users\\louislin\\OneDrive\\桌面\\data_analysis\\backtesting_system\\strategy\\macd_cross.json"
STRATEGY_NAME = "macd_cross"
class MACD_EMA(object):
    def __init__(self, data_paths: list, symbols: list):
        self.symbols = symbols
        self.data_paths = data_paths
        self.original_datas = self.loading_data()

        self.config = config(STRATEGY_CONFIG).load_config()[STRATEGY_NAME]
        self.param = self.config['param']
        self.limit = self.config['limit']
    def loading_data(self) -> dict:
        """Loads CSV files based on symbols and paths."""
        original_data = {}
        for i in range(len(self.data_paths)):
            try:
                original_data[self.symbols[i]] = pd.read_csv(self.data_paths[i])
            except FileNotFoundError as e:
                print(f"[Error] Unable to load file {self.symbols[i]} at {self.data_paths[i]}. Details: {e}")
        return original_data
    
    def generate_signal(self)->dict:
        """
        Generate signal
        return: dict
        {symbol(str), signal(pd.Dataframe)}
        """
        signals={}
        for symbol in self.symbols:
            data = self.original_datas[symbol]
            feature = pd.DataFrame()
            feature["long_ema"] = talib.EMA(data["Close"],150)
            rsi = talib.RSI(data['Close'])
            feature['Close'] = data['Close']
            feature['rsi'] = rsi 
            feature['rsi_shift1'] = rsi.shift(1)
            feature['rsi_shift2'] = rsi.shift(2)
            macd,signal,hist = talib.MACD(data["Close"])
            feature["hist"] = hist
            feature["diff"] = macd
            feature["dea"] = signal
            feature.to_csv("feature.csv")
            print("feature saved")
            def classify_signal(row: pd.Series) -> int:
                """ 計算交易訊號 """
                # 若滯後資料為空，回傳 0
                if pd.isnull(row["long_ema"]) or pd.isnull(row["dea"]) or pd.isnull(row["diff"])or pd.isnull(row["Close"]) or  pd.isnull(row["rsi"]):
                    return 0
                # up_stream = row['rsi']<30 and row['rsi_shift1']<30 #and row['rsi_shift2']<30
                # down_stream = row['rsi']>70 and row['rsi_shift1']>70# and row['rsi_shift2']>70
                if row["diff"]>row["dea"] and row['Close']>row['long_ema'] and row['rsi']>60:
                    return 1
                elif row["diff"]<row["dea"] and row['Close']<row['long_ema'] and row['rsi']<60:
                    return -1
                else:
                    return 0
            signals[symbol] = feature.apply(classify_signal, axis=1)
            signals[symbol].to_csv(f"{symbol}_signal.csv")
     
        
            

    
        return signals



# class MACD_CROSS(object):
#     def __init__(self, data_paths: list, symbols: list):
#         self.symbols = symbols
#         self.data_paths = data_paths
        
#         self.original_datas = self.loading_data()
        
#         self.config = config(STRATEGY_CONFIG).load_config()[STRATEGY_NAME]
#         self.param = self.config['param']
#         self.limit = self.config['limit']

#     def loading_data(self) -> dict:
#         """Loads CSV files based on symbols and paths."""
#         original_data = {}
#         for i in range(len(self.data_paths)):
#             try:
#                 original_data[self.symbols[i]] = pd.read_csv(self.data_paths[i])
#             except FileNotFoundError as e:
#                 print(f"[Error] Unable to load file {self.symbols[i]} at {self.data_paths[i]}. Details: {e}")
#         return original_data

#     def generate_signal(self) -> dict:
#         signals = {}

#         for symbol in self.symbols:
#             if symbol not in self.original_datas:
#                 raise ValueError(f"Market data for {symbol} not available.")
            
#             data = self.original_datas[symbol]
#             feature = pd.DataFrame()

#             # 計算 MACD 指標
#             indicator = MACD(data, "Close", 
#                             self.param['macd_short_period'], 
#                             self.param['macd_long_period'], 
#                             self.param['macd_signal_period'])

#             # 建立特徵 DataFrame
#             feature["diff"] = indicator.get_macd_line()
#             feature["dea"] = indicator.get_signal_line()
#             feature["long_ema"] = indicator.get_long_ema()
#             feature["mid_ema"] = ExponentialMovingAverage(data, "Close" , self.param['ema_period']).get_ema()
#             feature["short_ema"] = indicator.get_short_ema()
#             feature["moment"] = feature["diff"] - feature["dea"]

#             # 計算額外指標
#             feature["short_slope"] = feature["short_ema"].diff()
#             feature["mid_slope"] = feature["mid_ema"].diff()
#             feature["long_slope"] = feature["long_ema"].diff()

#             # 【修改處】：新增滯後欄位，避免在 row 上使用 shift()
#             feature["short_ema_prev"] = feature["short_ema"].shift(1)
#             feature["long_ema_prev"] = feature["long_ema"].shift(1)
#             feature["moment_prev"] = feature["moment"].shift(1)
#             feature["moment_prev2"] = feature["moment"].shift(2)

#             def classify_signal(row: pd.Series) -> int:
#                 """ 計算交易訊號 """
#                 # 若滯後資料為空，回傳 0
#                 if pd.isnull(row["short_ema_prev"]) or pd.isnull(row["moment_prev"]) or pd.isnull(row["moment_prev2"]):
#                     return 0

#                 ema_cross_up = (np.sign(row["short_ema"] - row["long_ema"]) > np.sign(row["short_ema_prev"] - row["long_ema_prev"])
#                                 and row["diff"] > 0)
#                 ema_cross_down = (np.sign(row["short_ema"] - row["long_ema"]) < np.sign(row["short_ema_prev"] - row["long_ema_prev"])
#                                 and row["diff"] < 0)

#                 threshold = 0.5
#                 moment_cross_up = (np.sign(row["moment"]) > np.sign(row["moment_prev"]) 
#                                 and abs(row["moment"] - row["moment_prev"]) > threshold)
#                 moment_cross_down = (np.sign(row["moment"]) < np.sign(row["moment_prev"]) 
#                                     and abs(row["moment"] - row["moment_prev"]) > threshold)

#                 uptrend = (row["short_ema"] > row["mid_ema"] > row["long_ema"] 
#                            and row["short_slope"] > 0 and row["mid_slope"] > 0)
#                 downtrend = (row["short_ema"] < row["mid_ema"] < row["long_ema"] 
#                              and row["short_slope"] < 0 and row["mid_slope"] < 0)

#                 if uptrend:
#                     if ema_cross_up and row["moment"] > 0:
#                         return 1
#                     elif ema_cross_down and moment_cross_down and row["diff"] < 0:
#                         return -1
#                     elif row["moment"] >= 0 and row["moment"] <= row["moment_prev"] and row["moment_prev"] <= row["moment_prev2"]:
#                         return -1
#                     elif row["moment"] <= 0:
#                         return 0
#                     else:
#                         return 0
#                 elif downtrend:
#                     if ema_cross_down and row["moment"] < 0:
#                         return -1
#                     elif moment_cross_down:
#                         return -1
#                     elif ema_cross_up and moment_cross_up and row["diff"] > 0:
#                         return 1
#                     elif ema_cross_up:
#                         if row["moment"] >= row["moment_prev"] and row["moment_prev"] >= row["moment_prev2"]:
#                             return 1
#                     else:
#                         return 0
#                 else:
#                     return 0  # 震盪區間
            
#             # 應用 classify_signal，並將結果儲存為一個 Series
#             signals[symbol] = feature.apply(classify_signal, axis=1)

#         return signals
# if __name__=="__main__":
#     symbol = ["BTCUSDT"]
#     data_path =[rf"C:\Users\louislin\OneDrive\桌面\data_analysis\backtesting_system\test.csv"]
#     macd = MACD_EMA(symbol,data_path)
#     macd.generate_signal()