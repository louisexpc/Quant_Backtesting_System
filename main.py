import pandas as pd

from utils.account import Account
from pkg.ConfigLoader import config
from strategy.macd_cross import MACD_CROSS
from utils.spot import Spot
from utils.data import DataSingleton

STRATEGY_CONFIG = r".\strategy\macd_cross.json"
STRATEGY_NAME = "macd_cross"

class backtest(object):
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
    
    def next(self):
        idx = self.ds.get_current_index()
        for symbol in self.symbols:
            if self.signals[symbol].iloc[idx]:
                current_price = self.spot.current_data[symbol]['Close']
                self.spot.create_order(symbol,500,take_profit=current_price*1.1,stop_loss=current_price*0.95)
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
        
        
if __name__ == '__main__':
    symbol = ["BTCUSDT"]
    data_path =[rf"C:\Users\louislin\OneDrive\桌面\data_analysis\backtesting_system\test.csv"]
    account = Account("test")
    account.spot_balance = 10000
    bt = backtest(data_path,symbol,account)
    bt.run()
