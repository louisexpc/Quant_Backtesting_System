import numpy as np
import pandas as pd
import random

from utils.account import Account
from utils.data import DataSingleton

COMMISSION = 0.1 * 0.01  # 0.1% 手續費 
MINIMUM_ORDER = 5        # 最小下單金額 5 USDT

class Spot:
    def __init__(self, account: Account):
        # 初始化持倉 DataFrame
        self.holdings = pd.DataFrame(columns=["symbol", "quantity", "avg_cost", "upnl"]).set_index("symbol")
        
        # 初始化訂單簿和交易記錄
        self.order_book = pd.DataFrame(columns=["order_id", "symbol", "side", "price", "quantity", "notional", "status", "timestamp"])
        self.trade_history = pd.DataFrame(columns=["trade_id", "symbol", "side", "price", "quantity", "notional", "timestamp", "datetime"])
        self._order_id = 0
        self.account = account
        # 初始化回測資料
        self.data_instance = DataSingleton([], [])
        # 註冊 callback，避免重複註冊由 DataSingleton 處理
        self.data_instance.register_callback(self.update_data)
        self.current_data = self.data_instance.get_current_data()
        self.frozen_balance = 0.0
        self.frozen_balance_quantity = {}
        # 新增淨值歷史紀錄列表，用於每個時間點儲存淨值
        self.net_value_history = []

    def update_data(self, idx):
        """
        當 DataSingleton 更新索引時觸發此 callback，
        更新市場數據、檢查掛單、重新計算未實現盈虧，
        並自動檢查是否需平倉。
        
        Parameters:
          idx: (回測時用，忽略即可)
        """
        self.current_data = self.data_instance.get_current_data()
        self.execute_order()
        self.calculate_upnl()
        # 計算並記錄每個時間點的淨值
        spot_balance = self.account.spot_balance
        holdings_value = sum(
            self.holdings.loc[symbol, "quantity"] * self.current_data[symbol]['Close']
            for symbol in self.holdings.index if symbol in self.current_data
        )
        net_value = spot_balance + holdings_value
        self.net_value_history.append(net_value)

    def calculate_fees(self, order_value: float) -> float:
        return order_value * COMMISSION

    def calculate_upnl(self):
        for symbol in self.holdings.index:
            current_price = self.current_data[symbol]['Close'] 
            quantity = self.holdings.loc[symbol, "quantity"]
            avg_cost = self.holdings.loc[symbol, "avg_cost"]
            upnl = (current_price - avg_cost) * quantity
            self.holdings.loc[symbol, "upnl"] = upnl

    def simulate_slippage(self, price, slippage_prob=0.001, slippage_range=(-0.005, 0.005)):
        if random.random() < slippage_prob:
            slippage = random.uniform(*slippage_range)
            return price * (1 + slippage)
        return price

    def create_buy_order(self, symbol: str, notional: float, price: float = None, type: str = "market"):
        """創建市價/限價買單
        parameters:
        -symbol:交易貨幣兌
        -notional:交易總價
        -price:限價單指定買入價格
        -type: market: 市價單, limit:限價單
        """
        if symbol not in self.current_data or self.current_data[symbol] is None:
            raise ValueError(f"Market data for {symbol} not available.")
        if self.account.spot_balance < notional:
            print(f"Insufficient balance (remain: {self.account.spot_balance})")
            return
        
        if type == "market":
            execution_price = self.simulate_slippage(self.current_data[symbol]['Close'])
            self._execute_buy_order(symbol, execution_price, "buy", notional, type)
        elif type == "limit":
            if price is None:
                raise ValueError("Limited order should have specific price")
            if price > self.current_data[symbol]['Close']:
                print(f"Limited price: {price} should be <= current price: {self.current_data[symbol]['Close']}")
                return
            self.account.spot_balance -= notional
            self.frozen_balance += notional
            order_record = {
                "order_id": self._order_id,
                "symbol": symbol,
                "side": "buy",
                "price": price,
                "notional": notional,
                "status": "pending",
                "type": "limit",
                "timestamp": self.data_instance.get_current_index(),
            }
            self.order_book = pd.concat([self.order_book, pd.DataFrame([order_record])], ignore_index=True)
            self._order_id += 1
        else:
            raise ValueError(f"Unsupported order type: {type}")

    def _execute_buy_order(self, symbol: str, execution_price: float, side: str, notional: float, type: str):
        if symbol not in self.current_data or self.current_data[symbol] is None:
            raise ValueError(f"Market data for {symbol} not available.")
        
        fees = self.calculate_fees(notional)
        net_notional = notional - fees
        quantity = net_notional / execution_price
        
        if type == "market":
            self.account.spot_balance -= notional
        elif type == "limit":
            self.frozen_balance -= notional
        
        # 更新持倉
        if symbol in self.holdings.index:
            current_quantity = self.holdings.loc[symbol, "quantity"]
            current_avg_cost = self.holdings.loc[symbol, "avg_cost"]
            new_quantity = current_quantity + quantity
            new_avg_cost = (current_avg_cost * current_quantity + execution_price * quantity) / new_quantity
            self.holdings.loc[symbol, "quantity"] = new_quantity
            self.holdings.loc[symbol, "avg_cost"] = new_avg_cost
        else:
            self.holdings.loc[symbol] = [quantity, execution_price, 0]
        
        # 記錄交易
        trade = {
            "trade_id": self._order_id,
            "symbol": symbol,
            "side": side,
            "price": execution_price,
            "quantity": quantity,
            "notional": net_notional,
            "timestamp": self.data_instance.get_current_index(),
            "datetime": self.data_instance.get_current_data()[symbol]['Datetime'],
        }
        self.trade_history = pd.concat([self.trade_history, pd.DataFrame([trade])], ignore_index=True)
        self._order_id += 1
        print(f"Buy\t executed: {symbol}, quantity: {quantity}, price: {execution_price}, notional: {net_notional}")

    def create_sell_order(self, symbol: str, quantity: float, price: float = None, type: str = "market"):
        if symbol not in self.current_data or self.current_data[symbol] is None:
            raise ValueError(f"Market data for {symbol} not available.")
        if symbol not in self.holdings.index or self.holdings.loc[symbol, "quantity"] < quantity:
            print(f"Insufficient quantity for sell: {symbol}, requested: {quantity}")
            return
        
        if type == "market":
            execution_price = self.simulate_slippage(self.current_data[symbol]['Close'])
            self._execute_sell_order(symbol, execution_price, "sell", quantity, "market")
        elif type == "limit":
            if price is None:
                raise ValueError("Limited order should have specific price")
            if price < self.current_data[symbol]['Close']:
                print(f"Limited price: {price} should be >= current price: {self.current_data[symbol]['Close']}")
                return
            self.holdings.loc[symbol, "quantity"] -= quantity
            self.frozen_balance_quantity[symbol] = self.frozen_balance_quantity.get(symbol, 0) + quantity
            order_record = {
                "order_id": self._order_id,
                "symbol": symbol,
                "side": "sell",
                "price": price,
                "quantity": quantity,
                "status": "pending",
                "type": "limit",
                "timestamp": self.data_instance.get_current_index(),
            }
            self.order_book = pd.concat([self.order_book, pd.DataFrame([order_record])], ignore_index=True)
            self._order_id += 1
        else:
            raise ValueError(f"Unsupported order type: {type}")

    def _execute_sell_order(self, symbol: str, execution_price: float, side: str, quantity: float, type: str):
        if symbol not in self.holdings.index:
            print(f"No holdings for {symbol}")
            return
        
        holding = self.holdings.loc[symbol]
        if holding["quantity"] < quantity:
            print(f"Insufficient quantity to sell: {symbol}, requested: {quantity}, available: {holding['quantity']}")
            return
        
        # 計算收益和手續費
        notional = execution_price * quantity
        fees = self.calculate_fees(notional)
        net_notional = notional - fees
        
        # 更新持倉
        new_quantity = holding["quantity"] - quantity
        if new_quantity > 0:
            self.holdings.loc[symbol, "quantity"] = new_quantity
        else:
            self.holdings.drop(symbol, inplace=True)
        
        # 更新帳戶餘額
        self.account.spot_balance += net_notional
        
        # 記錄交易
        trade = {
            "trade_id": self._order_id,
            "symbol": symbol,
            "side": side,
            "price": execution_price,
            "quantity": quantity,
            "notional": net_notional,
            "timestamp": self.data_instance.get_current_index(),
            "datetime": self.data_instance.get_current_data()[symbol]['Datetime'],
        }
        self.trade_history = pd.concat([self.trade_history, pd.DataFrame([trade])], ignore_index=True)
        self._order_id += 1
        print(f"Sell\t executed: {symbol}, quantity: {quantity}, price: {execution_price}, notional: {net_notional}")

    def execute_order(self):
        for idx, order in self.order_book.iterrows():
            if order["status"] != "pending":
                continue
            
            symbol = order["symbol"]
            current_price = self.current_data[symbol]['Close']
            
            if order["side"] == "buy" and current_price <= order["price"]:
                self._execute_buy_order(symbol, order["price"], "buy", order["notional"], "limit")
                self.order_book.at[idx, "status"] = "filled"
            elif order["side"] == "sell" and current_price >= order["price"]:
                self._execute_sell_order(symbol, order["price"], "sell", order["quantity"], "limit")
                self.order_book.at[idx, "status"] = "filled"

    def evolution(self, Rf: float = 0.0):
        """
        評估交易績效，返回 Sharpe Ratio、PLR、Win Rate、MDD 和 ROI。
        
        Parameters:
            Rf: 無風險利率，預設為 0.0
        
        Returns:
            dict: 包含各項指標的數值
        """
        # 檢查 net_value_history 是否有足夠資料，而不是僅依賴 trade_history
        if len(self.net_value_history) < 2:
            return {
                "sharpe_ratio": None,
                "profit_loss_ratio": None,
                "win_rate": None,
                "maximum_drawdown": None,
                "roi": None
            }
        
        # 使用提前記錄的 net_value_history，而不是即時計算
        net_values = pd.Series(self.net_value_history)
        returns = net_values.pct_change().dropna()  # 計算回報率並移除 NaN
        
        # 1. Sharpe Ratio
        # 增加標準差為零的檢查，避免除以零錯誤
        if len(returns) > 1 and returns.std() != 0:
            sharpe_ratio = (returns.mean() - Rf) / returns.std()
        else:
            sharpe_ratio = None
        
        # 2. 勝率 (Win Rate)
        if not returns.empty:
            win_rate = (returns > 0).mean()
        else:
            win_rate = None
        
        # 3. 盈虧比 (Profit/Loss Ratio)
        gains = returns[returns > 0]  # 正收益
        losses = -returns[returns < 0]  # 負收益取絕對值
        if not gains.empty and not losses.empty:
            avg_gain = gains.mean()
            avg_loss = losses.mean()
            plr = avg_gain / avg_loss if avg_loss != 0 else float('inf')
        else:
            plr = None
        
        # 4. 最大回撤 (Maximum Drawdown)
        peak = net_values.cummax()  # 計算歷史最高點
        drawdown = (net_values - peak) / peak  # 計算回撤
        mdd = drawdown.min()  # 最大回撤
        
        # 5. 投資回報率 (ROI)
        initial_balance = self.net_value_history[0]
        final_net_value = net_values.iloc[-1]
        roi = (final_net_value - initial_balance) / initial_balance if initial_balance != 0 else 0
        
        # 返回結果
        return {
            "sharpe_ratio": sharpe_ratio,
            "profit_loss_ratio": plr,
            "win_rate": win_rate,
            "maximum_drawdown": mdd,
            "roi": roi
        }
    
