import pandas as pd

from utils.account import Account
from pkg.ConfigLoader import config
from strategy.macd_cross import MACD_CROSS
from utils.spot import Spot
from utils.future import Future
from utils.data import DataSingleton
from utils.indicator import ATR

STRATEGY_CONFIG = r".\strategy\macd_cross.json"
STRATEGY_NAME = "macd_cross"

class backtest_spot(object):
    def __init__(self,data_paths:list,symbols:list,account:Account):
        self.data_paths = data_paths
        self.symbols = symbols
        self.ds = DataSingleton(data_paths,symbols)
        self.account = account
        self.spot = Spot(account)
        
        """ Init MACD Cross Strategy """
        self.config = config(STRATEGY_CONFIG).load_config()[STRATEGY_NAME]
        self.limit = self.config['limit']
        self.param = self.config['param']
        self.signals = MACD_CROSS(data_paths,symbols).generate_signal()
        self.atrs = self.genereate_ATR()
        
    def genereate_ATR(self)->dict:
        atrs = {}
        for symbol in self.symbols:
            data = self.ds.original_datas[symbol]
            atr = ATR(data).get_atr()
            atrs[symbol] = atr
            #atr.to_csv("test_atr.csv")
        return atrs
    
    def next(self):
        idx = self.ds.get_current_index()
        threshold = 2
        for symbol in self.symbols:
            if self.signals[symbol].iloc[idx]==1:
                current_price = self.spot.current_data[symbol]['Close']
                atr_range = threshold * self.atrs[symbol].iloc[idx]
                take_profit = current_price + atr_range
                stop_loss = current_price - atr_range
                self.spot.create_order(symbol,500,take_profit=take_profit,stop_loss=stop_loss)
    def run(self):
        print(f"Init: {self.ds.get_current_index()} and {self.ds.is_finished()}")
        for i in range(self.ds.get_total_rows()):
            print(self.ds.get_current_index())
         
            self.next()
            #print(f"day {i}:\n{self.spot.current_data}")
            #print(f"balance:\n{self.account.spot_balance}")
            #print(f"order_book:\n{self.spot.order_book}")
            #print(f"position:\n{self.spot.positions}")
            self.ds.run()
        """Final 結算"""
        total_balance = self.account.spot_balance
        if not self.spot.positions.empty:
            print(f"持有倉位:\n{self.spot.positions}")
            for _,pos in self.spot.positions.iterrows():
                current_price = self.ds.get_current_data()[pos['symbol']]['Close']
                total_balance+= (float(pos['size']) * float(current_price))

        if not self.spot.history_positions.empty:
            print(f"歷史交易紀錄:\n{self.spot.history_positions}")
            self.spot.history_positions.to_csv("history.csv")
        print(f"結餘:{total_balance}")

        evolution = self.spot.evolution()
        for symbol,indicators in evolution.items():
            print(f"標的資產: {symbol}")
            for indicator,value in indicators.items():
                print(f"{indicator}: {value}")
        #print(f"盈虧:{self.spot.history_positions['pnl'].sum()}")
        
class backtest_future:
    def __init__(self, data_paths: list, symbols: list, account: Account):
        self.data_paths = data_paths
        self.symbols = symbols
        self.ds = DataSingleton(data_paths, symbols)
        self.account = account
        self.future = Future(account)
        
        """ Init MACD Cross Strategy """
        self.config = config(STRATEGY_CONFIG).load_config()[STRATEGY_NAME]
        self.limit = self.config['limit']
        self.param = self.config['param']  # 目前尚未使用
        
        self.signals = MACD_CROSS(data_paths, symbols).generate_signal()
        self.atrs = self.generate_ATR() 

    def generate_ATR(self) -> dict: 
        """ 生成ATR資料 """
        atrs = {}
        for symbol in self.symbols:
            data = self.ds.original_datas[symbol]
            atr = ATR(data).get_atr()
            atrs[symbol] = atr
            # atr.to_csv("test_atr.csv")
        return atrs
    
    def atr_trailing_stop(self):
        """棘輪止損法（ATR Trailing Stop） 動態移動止損
        當浮動盈利達到 1 倍 ATR 時，選擇一個近期小低點作為參考點。
        根據持倉天數，每天上調止損 0.05 倍 ATR。
        """
        if self.future.positions.empty:
            return

        def calculate_stop_loss(pos: pd.Series):
            # 若pnl達到take_profit的0.667(約2/3)時才提高止損
            if "pnl" in pos.index and pos["pnl"] is not None and float(pos["pnl"]) >= float(pos["take_profit"]) * 0.667:
                idx = self.ds.get_current_index()
                hold_pos_day = idx - pos["timestamp"]
                # 根據註解，假設想要每天上調 0.05 * ATR，可做以下修正:
                # pos["stop_loss"] += hold_pos_day * 0.05 * self.atrs[pos["symbol"]].iloc[idx]
                # 若仍要維持原邏輯(直接整倍數乘上天數)，可保留:
                if hold_pos_day != 0:
                    pos["stop_loss"] += hold_pos_day * 0.05 * self.atrs[pos["symbol"]].iloc[idx]
            return pos  # 修改：回傳 row，才能更新原 DataFrame

        # 修改：需將 apply 結果回存
        self.future.positions = self.future.positions.apply(calculate_stop_loss, axis=1)
    
    def atr_trailing_take_profit(self):
        """
        當價格達到 1 倍 ATR 的盈利時，根據 ATR 棘輪法調整止盈，以捕捉更大趨勢。
        """
        if self.future.positions.empty:
            return
        
        def calculate_take_profit(pos: pd.Series):
            if "pnl" in pos.index and pos["pnl"] is not None and float(pos["pnl"]) >= float(pos["take_profit"]) * 0.667:
                idx = self.ds.get_current_index()
                hold_pos_day = idx - pos["timestamp"]
                new_take_profit = pos["entry_price"] + hold_pos_day * self.atrs[pos["symbol"]].iloc[idx]
                if hold_pos_day != 0 and new_take_profit < (
                    pos["entry_price"] + 2 * (self.atrs[pos["symbol"]].iloc[pos["timestamp"]])
                ):
                    pos["take_profit"] = new_take_profit
            return pos  # 修改：回傳 row

        # 修改：需將 apply 結果回存
        self.future.positions = self.future.positions.apply(calculate_take_profit, axis=1)

    def next(self):
        idx = self.ds.get_current_index()
        
        for symbol in self.symbols:
            current_price = self.future.current_data[symbol]['Close']
            atr = self.atrs[symbol].iloc[idx]
            if self.signals[symbol].iloc[idx] != 0:
                # 根據 signal 下單 side = 1 多單 side = -1 空單
                side = self.signals[symbol].iloc[idx]
                take_profit = current_price + side * atr * 1.5
                stop_loss = current_price - side * 1
                self.future.create_order(symbol, side, 1000, take_profit=take_profit, stop_loss=stop_loss)
        
        self.atr_trailing_stop()
        self.atr_trailing_take_profit()

    def run(self):
        self.ds.reset()
        print(f"Init: {self.ds.get_current_index()} and {self.ds.is_finished()}")
        
        for i in range(self.ds.get_total_rows()):
            
            self.next()
            self.ds.run()
        
        print(f"Backtest Down, remain: {self.account.future_balance}")
        
        if not self.future.positions.empty:
            print(f"持有倉位:\n{self.future.positions}")
            # 修改：更正檔名拼字
            self.future.positions.to_csv("future_positions.csv")  # 原為 future_postions.csv
        
        if not self.future.history_positions.empty:
            print(f"歷史交易紀錄:\n{self.future.history_positions}")
            self.future.history_positions.to_csv("history.csv")
        
        evolution = self.future.evolution()
        for symbol, indicators in evolution.items():
            print(f"標的資產: {symbol}")
            for indicator, value in indicators.items():
                print(f"{indicator}: {value}")

  
            

if __name__ == '__main__':
    symbol = ["BTCUSDT"]
    data_path =[rf"C:\Users\louislin\OneDrive\桌面\data_analysis\backtesting_system\data\4h_BTCUSDT.csv"]
    account = Account("test")
    account.set_spot_balance(10000)
    account.set_future_balance(10000)
    # bt = backtest_spot(data_path,symbol,account)
    # bt.run()
    bt = backtest_future(data_path,symbol,account)
    bt.run()