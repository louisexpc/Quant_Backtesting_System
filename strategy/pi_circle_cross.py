import pandas as pd
import numpy as np
import os
import time
import talib

from pkg.ConfigLoader import config

class PI_CIRCLE(object):
    def __init__(self, data_paths: list, symbols: list):
        self.symbols = symbols
        self.data_paths = data_paths
        self.original_datas = self.loading_data()

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
        """Compute PI circle"""
        signals={}
        for symbol in self.symbols:
            data = self.original_datas[symbol]
            feature = pd.DataFrame()
            feature["111EMA"] = talib.DEMA(data['Close'],timeperiod=111)
            feature['350EMA_']=talib.DEMA(data['Close'],timeperiod=350)
            feature["350EMA"] = talib.DEMA(data['Close'],timeperiod=350)
            feature.to_csv(f"{symbol}__PI_Circle_feature.csv")
            def classify_signal(row: pd.Series) -> int:
                """ 計算交易訊號 """
                if pd.isnull(row["111EMA"]) or pd.isnull(row["350EMA"]):
                    return 0
                if row['111EMA']>row['350EMA']:
                    return -1
                elif row['111EMA']<row['350EMA']:
                    return 1
            signals[symbol] = feature.apply(classify_signal, axis=1)
            signals[symbol].to_csv(f"{symbol}_PI_Circle_signal.csv")
        return signals
                