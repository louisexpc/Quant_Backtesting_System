import pandas as pd
from datetime import datetime, timedelta
import time

from utils.price_crawler import fetch_all_ohlcv
from utils.account import Account
from pkg.ConfigLoader import config
from strategy.macd_cross import MACD_EMA
from strategy.pi_circle_cross import PI_CIRCLE
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
        
        self.signals = MACD_EMA(data_paths, symbols).generate_signal()
    
    def next(self):
        idx = self.ds.get_current_index()
        
        for symbol in self.symbols:
            if self.signals[symbol].iloc[idx]==1 and self.account.spot_balance>=5000:
                self.spot.create_buy_order(symbol,5000)
            elif self.signals[symbol].iloc[idx] == -1 and symbol in self.spot.holdings.index:
                self.spot.create_sell_order(symbol,self.spot.holdings.loc[symbol]['quantity'])
    def run(self):
        print(f"Init:{self.ds.is_finished()}")
        for i in range(self.ds.get_total_rows()):
          
         
            self.next()
            self.ds.run()

        # 最終結算
        total_balance = self.account.spot_balance
        if not self.spot.holdings.empty:
            print(f"持有倉位:\n{self.spot.holdings}")
            for symbol in self.spot.holdings.index:
                current_price = self.ds.get_current_data()[symbol]['Close']
                total_balance += float(self.spot.holdings.loc[symbol, 'quantity']) * float(current_price)

        if not self.spot.trade_history.empty:
            print(f"歷史交易紀錄:\n{self.spot.trade_history}")
            self.spot.trade_history.to_csv("history.csv")
        print(f"結餘: {total_balance}")

        # 交易績效評估
        evolution = self.spot.evolution()  # 使用 evaluate_performance
        print("交易績效評估結果：")
        for metric, value in evolution.items():
            print(f"{metric}: {value}")

        # 計算總盈虧
        if not self.spot.trade_history.empty:
            total_pnl = self.spot.trade_history['notional'].sum()
            print(f"總盈虧: {total_pnl}")

class backtest_future_base(object):
    def __init__(self, data_paths: list, symbols: list, account: Account):
        """
        Augments:
        - data_paths: List of file paths to the market data CSVs.
            - each csv must contain at least ['Timestamp','Open','High','Low','Close','Volume'] columns (Upper / Lower case are all allowed).
        - symbols: List of trading symbols corresponding to the data paths.
        - account: An instance of the Account class to manage balances and positions.
        """
        self.data_paths = data_paths
        self.symbols = symbols
        self.ds = DataSingleton(data_paths, symbols)
        self.account = account
        self.future = Future(account)
        
        """ Init Your Strategy """
    def next(self):
        idx = self.ds.get_current_index()
        for symbol in self.symbols:
            """ Implement your strategy logic here """
            pass


    def run(self):
        self.ds.reset()
        print(f"Init: {self.ds.get_current_index()} and {self.ds.is_finished()}")
        
        for i in range(self.ds.get_total_rows()):
            
            self.next()
            self.ds.run()
        total_future_balance = self.account.future_balance
        
        
        if not self.future.positions.empty:
            print(f"持有倉位:\n{self.future.positions}")
            for _,pos in self.future.positions.iterrows():
                if "unrealized_pnl" in pos.index:
                    total_future_balance+=float(pos['unrealized_pnl'])
                total_future_balance+=float(pos.get('margin'))
            self.future.positions.to_csv("future_positions.csv")  #原為 future_postions.csv
        else:
            print("Not holding position")
        
        if not self.future.history_positions.empty:
            print(f"歷史交易紀錄:\n{self.future.history_positions}")
            self.future.history_positions.to_csv("history.csv")
        else:
            print("Not any trading")
        print(f"Backtest Down, remain: {total_future_balance}")
        evolution = self.future.evolution()
        for symbol, indicators in evolution.items():
            print(f"標的資產: {symbol}")
            for indicator, value in indicators.items():
                print(f"{indicator}: {value}")

class SNR_backtest_future(backtest_future_base):
    def __init__(self, data_paths: list, symbols: list, account: Account, signal_path:str):
        """
        Augments:
        - data_paths: List of file paths to the market data CSVs.
            - each csv must contain at least ['Timestamp','Open','High','Low','Close','Volume'] columns (Upper / Lower case are all allowed).
        - symbols: List of trading symbols corresponding to the data paths.
        - account: An instance of the Account class to manage balances and positions.
        """
        super().__init__(data_paths, symbols, account)
        """ Init SNR signals df with t0/t1 : entry/exit signals and side column """
        self.signals_df = pd.read_csv(signal_path)
        self.signals_df['t0'] = pd.to_datetime(self.signals_df['t0'])
        self.signals_df['t1'] = pd.to_datetime(self.signals_df['t1'])
        
        self.signals_df['side'] = self.signals_df['side'].astype(int)
        self.signals_df = self.signals_df[self.signals_df['side'] == 1]
        # self.signals_df = self.signals_df[self.signals_df['t0'] > pd.to_datetime("2025-04-30 18:00:00+08:00")]
        # print(f"[Debug] {self.signals_df.head()}")
        if 'pred_vote' in self.signals_df.columns:
            self.signals_df['pred_vote'] = self.signals_df['pred_vote'].astype(int)

        self.signals_df.reset_index(drop=True, inplace=True)
        self.signals = {}


    def next(self):
        idx = self.ds.get_current_index()
        
        for symbol in self.symbols:
            
            current_price = self.future.current_data[symbol]['Close']
            current_time = pd.to_datetime(self.ds.original_datas[symbol].iloc[idx]['Datetime'])


            if self.signals_df['t0'].isin([current_time]).any():
                # print(f"[Debug] Entry signal for {symbol} at {current_time}")
                # 根據 signal 下單 side = 1 多單 side = -1 空單
                signal_row = self.signals_df[self.signals_df['t0'] == current_time].iloc[0]
                side = signal_row['side']

                if 'pred_vote' in signal_row and signal_row['pred_vote'] != 1:
                    continue
                # 下單: 計算 leverage 後下單
                # 預期 tp/sl 是 3.6倍 的 signal volatility
                # 預期收益是 10% 的 position size(default 1000)
                # leverage = (0.1 * 1000) / (3.6 * signal_row['volatility'])
                pt = float(signal_row['pt'])
                sl = float(signal_row['sl'])

                position_size = 1000
                leverage = side * 0.1 / ((pt-current_price)/current_price)
                leverage = int(max(1, min(leverage, 50)))  # 限制在 1 到 50 倍之間
                # print(f"[Debug] {symbol} leverage: {leverage}")

                self.future.create_order(symbol, side, position_size, leverage=leverage, take_profit=pt, stop_loss=sl)
            # else:
            #     print(f"[Debug] No entry signal for {symbol} at {current_time}")


class macd_backtest_future(backtest_future_base):
    def __init__(self, data_paths: list, symbols: list, account: Account):
        """
        Augments:
        - data_paths: List of file paths to the market data CSVs.
            - each csv must contain at least ['Timestamp','Open','High','Low','Close','Volume'] columns (Upper / Lower case are all allowed).
        - symbols: List of trading symbols corresponding to the data paths.
        - account: An instance of the Account class to manage balances and positions.
        """
        super().__init__(data_paths, symbols, account)
        """ Init MACD Cross Strategy """
        self.config = config(STRATEGY_CONFIG).load_config()[STRATEGY_NAME]
        self.limit = self.config['limit']
        self.param = self.config['param']  
        
        self.signals = MACD_EMA(data_paths, symbols).generate_signal()
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



  
            

if __name__ == '__main__':
    # symbols = ["BTCUSDT"]  # 交易對

    # timeframe = '1d'     # 時間週期
    # since = int(datetime(2025, 1, 1).timestamp() * 1000)  # 起始時間（yyyy-mm-dd）

    # # 獲取完整資料
    # data = fetch_all_ohlcv(symbols, timeframe, since)
    # symbol = ["BTCUSDT"]
    # timeframe = '15m'
    # since = int(datetime(2024, 1, 1).timestamp() * 1000)  # 起始時間（yyyy-mm-dd）
    # data_path = fetch_all_ohlcv(symbol,timeframe,since)
    
    # start = datetime.now()
    # account = Account("test")
    # account.set_spot_balance(10000)
    # #account.set_future_balance(1000000)
    # bt = backtest_spot(data_path,symbol,account)
    # # bt.run()
    # #bt = backtest_future(data_path,symbol,account)
    # bt.run()
    # print(f"RUN TIME:{datetime.now()-start}")

    """SNR Backtest Future"""
    symbols = ["BTCUSDT"]  # 交易對

    account = Account("test")
    account.set_future_balance(1000000)
    bt = SNR_backtest_future(
        data_paths=["./data/binanceusdm_swap_BTC-USDT-USDT_1h.csv"],
        symbols=symbols, 
        account=account,
        signal_path="./data/BTC-USDT_1h_ewma_up3_dn3_lookback36_label.csv"
    )
    bt.run()