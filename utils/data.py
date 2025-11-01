# utils/data.py
import pandas as pd
import numpy as np

class DataSingleton:
    '''
    初始化回測資料集。
    Parameters:
    symbols: 回測所需要的貨幣對資料, dtype = list
    data_paths: 貨幣對資料csv檔存放路徑, dtype = list

    Hint: 所有資料須保證等長(row)，並且已經清理過無缺失
    '''
    _instance = None

    def __new__(cls, *args, **kwargs):
        """Singleton implementation to ensure a single data instance."""
        if cls._instance is None:
            cls._instance = super(DataSingleton, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, data_paths: list, symbols: list):
        """Loads data for backtesting and initializes event-driven framework."""
        if not self._initialized:
            self.symbols = symbols
            self.data_paths = data_paths
            self._idx = 0  # 統一的時間索引
            self.length = 0
            self.original_datas = self.loading_data()
            self.callbacks = []  # 註冊所有在 _idx 更新時要觸發的 callback 函數
            self._initialized = True

    def loading_data(self) -> dict:
        """Loads CSV files based on symbols and paths."""
        original_data = {}
        for i in range(len(self.data_paths)):
            try:
                df = pd.read_csv(self.data_paths[i])
                """
                Column Standardization:
                - 先將所有欄位轉為大寫 
                - 確保所有欄位包含: ['Datetime','Open','High','Low','Close','Volume']
                """
                df.columns = [col.capitalize() for col in df.columns]
                required_cols = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_cols:
                    if col not in df.columns:
                        raise ValueError(f"Missing required column '{col}' in data for symbol {self.symbols[i]}")
                df = df[required_cols]
                
                original_data[self.symbols[i]] = df
                # 【修改處】：以 len(df) 計算行數，取最小值
                if self.length == 0:
                    self.length = len(df)
                else:
                    self.length = min(self.length, len(df))
            except FileNotFoundError as e:
                print(f"[Error] Unable to load file {self.symbols[i]} at {self.data_paths[i]}. Details: {e}")
        return original_data

    def register_callback(self, callback):
        """Registers a function that will be called whenever _idx is updated."""
        if callable(callback) and callback not in self.callbacks:
            self.callbacks.append(callback)

    def run(self):
        """
        Advances _idx and triggers all registered callbacks.
        【修改處】：若已到最後一筆資料則不再增加 _idx，
        並可以選擇性回傳最後一筆資料供後續總結回測成果。
        """
        if self._idx < self.length - 1:
            self._idx += 1
            for callback in self.callbacks:
                callback(self._idx)
        else:
            # 已到最後一筆，回傳最後一筆資料
            for callback in self.callbacks:
                callback(self._idx)
            # 或可加入提醒訊息
            print(f"Reached the last row of data.id: {self._idx}")


    def get_current_data(self) -> dict:
        """Returns the current row of data for backtesting."""
        return {
            symbol: (df.iloc[self._idx] if self._idx < len(df) else None)
            for symbol, df in self.original_datas.items()
        }

    def get_current_index(self) -> int:
        """Returns the current index (time step)."""
        return self._idx

    def get_total_rows(self) -> int:
        """【新增功能】Returns the total number of rows in the data."""
        return self.length

    def is_finished(self) -> bool:
        """【新增功能】Returns True if the current index is at the last row."""
        return self._idx >= self.length - 1

    def reset(self):
        """Resets the index for a new backtesting session."""
        self._idx = 0

if __name__ =='__main__':
    symbols = ["BTCUSDT",""]
    data_paths = [r"C:\Users\louislin\OneDrive\桌面\data_analysis\backtesting_system\4h_BTCUSDT.csv"]
    
    ds = DataSingleton(data_paths, symbols)
    print("DataSingleton 初始化完成，載入市場數據：")
    print(ds.original_datas)
    print(ds.get_current_index())
    print(ds.get_current_data())
