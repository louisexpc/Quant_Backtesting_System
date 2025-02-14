import numpy as np
import pandas as pd
import random

from utils.account import Account
from utils.data import DataSingleton

COMMISSION = 0.1 * 0.01  # 0.1% 手續費 
MINIMUM_ORDER = 5        # 最小下單金額 5 USDT

class Spot:
    """
    現貨交易模擬系統
    Attributes:
      - history_positions: 關閉倉位的紀錄 (DataFrame)
      - positions: 當前持倉 (DataFrame)
      - order_book: 待成交的限價單記錄 (DataFrame)
      - _order_id: 訂單流水號
      - account: 帳戶資訊 (Account 物件)
      - data_instance: DataSingleton 取得市場數據
      - current_data: 最新市場數據 snapshot
    """
    def __init__(self, account: Account):
        """
        Initialize 現貨交易模擬系統.
        訂單格式包含：
          - order_id: 訂單 ID
          - symbol: 交易標的
          - entry_price: 買入價格
          - size: 實際成交的數量（標的資產數量）
          - timestamp: 建倉時的市場索引
          - take_profit: (可選) 限價止盈價格
          - stop_loss: (可選) 限價止損價格
          - margin: 保證金
        """
        self.history_positions = pd.DataFrame()
        self.positions = pd.DataFrame()
        self.order_info = ["order_id", "symbol", "entry_price", "size", "timestamp", "take_profit", "stop_loss"]
        self.order_book = pd.DataFrame(
            columns=["order_id", "symbol", "price", "size", "status", "take_profit", "stop_loss", "margin"]
        )
        self._order_id = 0
        self.account = account
        self.data_instance = DataSingleton([], [])
        #print(f"initial future data instance:\n{self.data_instance.get_current_index()}\n{self.data_instance.get_current_data()}")
        # 註冊 callback，避免重複註冊由 DataSingleton 處理
        self.data_instance.register_callback(self.update_data)
        # 初始化時直接取得市場資料，避免 self.current_data 為空
        self.current_data = self.data_instance.get_current_data()
    
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
        self.calculate_unrealized_pnl()
        self.close_position()
    
    def calculate_fees(self, order_value: float) -> float:
        """
        計算交易手續費: 最終成交額 = 訂單總價 - 手續費
        Parameters: 
          order_value: 訂單總價值 (USDT)
        Return:
          手續費 (USDT)
        """
        return order_value * COMMISSION
    
    def calculate_unrealized_pnl(self):
        """計算未實現盈虧 (UPNL) 並更新持倉資料"""
        if self.positions.empty:
            return

        def compute_upnl(row: pd.Series):
            symbol = row["symbol"]
            if symbol in self.current_data and self.current_data[symbol] is not None:
                current_price = float(self.current_data[symbol]['Close'])
                return (current_price - float(row['entry_price'])) * float(row['size'])
            return 0.0

        self.positions["unrealized_pnl"] = self.positions.apply(compute_upnl, axis=1)
    
    def simulate_slippage(self, price, slippage_prob=0.001, slippage_range=(-0.005, 0.005)):
        """模擬滑價效應"""
        if random.random() < slippage_prob:
            slippage = random.uniform(*slippage_range)
            return price * (1 + slippage)
        return price
    
    def create_order(self, symbol: str, notional: float,
                     price: float = None, order_type="market", 
                     take_profit: float = None, stop_loss: float = None):
        """
        創建訂單，支援市價單、限價單，並可附帶限價止盈與止損。
        此處 notional 表示訂單總價值 (USDT)
        
        下單前會檢查帳戶現金餘額是否足夠。
        
        Parameters:
          - symbol: 交易標的
          - notional: 欲成交訂單總價 (USDT)
          - price: 限價單時必填 (USDT)
          - order_type: "market" 或 "limit"
          - take_profit: (可選) 限價止盈價格 (USDT)
          - stop_loss: (可選) 限價止損價格 (USDT)
        """
        if symbol not in self.current_data or self.current_data[symbol] is None:
            raise ValueError(f"Market data for {symbol} not available.")
        
        if self.account.spot_balance < notional:
            #raise ValueError(f"Insufficient funds to buy spot {symbol} (remain balance:{self.account.spot_balance}).")
            print(f"Insufficient funds to buy spot {symbol} (remain balance:{self.account.spot_balance}).")
            return
        
        # 市價單：使用當前成交價計算實際數量，手續費於成交時扣除
        if order_type == "market":
            execution_price = float(self.simulate_slippage(self.current_data[symbol]['Close']))
            # 此處不在 create_order 中扣款，扣款邏輯放在 _execute_order 中
            self._execute_order(symbol, notional, execution_price, take_profit, stop_loss)
        # 限價單：立即扣除 notional 並放入掛單簿
        elif order_type == "limit":
            if price is None:
                raise ValueError("Price must be provided for limit orders.")
            price = float(price)
            self.account.spot_balance -= notional  # 立即扣除買入金額
            order_record = {
                "order_id": self._order_id,
                "symbol": symbol,
                "price": price,
                "order_value": notional,
                "status": "pending",
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "margin": notional  # 現貨交易 margin 即為 notional
            }
            self.order_book = pd.concat([self.order_book, pd.DataFrame([order_record])], ignore_index=True)  # 【修改處】
            self._order_id += 1
        else:
            raise ValueError(f"Unsupported order type: {order_type}")
    
    def _execute_order(self, symbol: str, notional: float, execution_price: float,
                       take_profit: float = None, stop_loss: float = None, reserved_margin: float = None):
        """
        內部方法：執行訂單，計算手續費，並建立持倉部位，同時儲存止盈與止損價格（若有設定）。
        """
        fees = self.calculate_fees(notional)
        net_notional = notional - fees
        quantity = net_notional / execution_price
        
        if net_notional < MINIMUM_ORDER:
            raise ValueError(f"Order value must be at least {MINIMUM_ORDER} USDT.")
        # 市價單：若 reserved_margin 未提供則從現金扣除 notional一次【修改處】
        if reserved_margin is None:
            if self.account.spot_balance < notional:
                raise ValueError(f"Insufficient funds to buy spot {symbol} (remain balance:{self.account.spot_balance}).")
            self.account.spot_balance -= notional  # 扣除買入金額
        order = {
            "order_id": self._order_id,
            "symbol": symbol,
            "entry_price": execution_price,
            "size": quantity,
            "timestamp": self.data_instance.get_current_data()[symbol]['Datetime'],
            "take_profit": take_profit,
            "stop_loss": stop_loss
        }
        print(f"Buy:\n{order}\nremain: {self.account.spot_balance}")
        self._order_id += 1
        self.positions = pd.concat([self.positions, pd.DataFrame([order])], ignore_index=True)
       
    
    def execute_order(self):
        """
        處理 order_book 中所有待成交的限價單：
          - 當前價格 <= 限價單價格時成交；
        成交後呼叫 _execute_order 並從掛單中移除該訂單。
        """
        if self.order_book.empty:
            return
        
        def check_and_execute(order: pd.Series):
            symbol = order["symbol"]
            if symbol not in self.current_data or self.current_data[symbol] is None:
                return False
            current_price = float(self.current_data[symbol]['Close'])
            if current_price <= order["price"]:
                reserved_margin = order.get("margin")
                self._execute_order(symbol, order["order_value"], order["price"],
                                    take_profit=order.get("take_profit"),
                                    stop_loss=order.get("stop_loss"),
                                    reserved_margin=reserved_margin)
                return True
            return False
        
        executed_orders = self.order_book.apply(check_and_execute, axis=1)
        self.order_book.drop(self.order_book[executed_orders].index, inplace=True)
    
    def close_position(self, order_id: int = None):
        """
        平倉作業：
          - 若傳入 order_id，則手動賣出該部位；
          - 否則，自動檢查所有持倉，若觸發止盈／止損條件則賣出部位。
          
        自動平倉條件：
            - 止盈條件：若設定 take_profit 且當前價格 >= take_profit
            - 止損條件：若設定 stop_loss 且當前價格 <= stop_loss
          
        平倉後計算盈虧、扣除手續費，將部位移至 history_positions，並將盈虧回補到現金帳戶中。
        Returns:
          平倉部位的 DataFrame
        """
        if self.positions.empty:
            return None
        if order_id is not None:
            position = self.positions[self.positions["order_id"] == order_id]
            if position.empty:
                return None
        else:
            def close_condition(row: pd.Series):
                symbol = row["symbol"]  
                if symbol not in self.current_data or self.current_data[symbol] is None:
                    return False
                current_price = float(self.current_data[symbol]["Close"])
                tp_triggered = (pd.notnull(row.get("take_profit")) and current_price >= row["take_profit"])
                sl_triggered = (pd.notnull(row.get("stop_loss")) and current_price <= row["stop_loss"])
                return tp_triggered or sl_triggered
            position = self.positions[self.positions.apply(close_condition, axis=1)]
        
        if position.empty:
            return None
        
        closed_positions = []
        for _, pos in position.iterrows():
            symbol = pos["symbol"]
            if symbol not in self.current_data or self.current_data[symbol] is None:
                continue
            exit_price = float(self.simulate_slippage(self.current_data[symbol]["Close"]))
            
            gain = exit_price * float(pos["size"])
            fees = self.calculate_fees(abs(gain))
            gain -= fees
            pnl = gain - pos["size"]*pos["entry_price"]
            # 將盈虧回補到現金帳戶中
            self.account.spot_balance += gain
            closed_position = pos.to_dict()
            closed_position["exit_price"] = exit_price
            closed_position["exit_timestamp"] = self.data_instance.get_current_data()[symbol]['Datetime']
            closed_position["gain"] = gain
            closed_position["pnl"] = pnl
            closed_positions.append(closed_position)
        self.positions = self.positions.drop(position.index)
        self.history_positions = pd.concat([self.history_positions, pd.DataFrame(closed_positions)], ignore_index=True)
        print("Sell:\n")
        for order in closed_positions:
            print(f"{order}\n remain:{self.account.spot_balance}")
        return pd.DataFrame(closed_positions)
        
    def evolution(self, Rf: float = 0.0) -> dict:
        """
        計算並回傳回測期間的指標評估，支援多標的：
        - Sharpe Ratio: (E(Rp)-Rf) / σp, 其中 Rp = net_profit / total_cost
        - Profit/Loss Ratio: 平均獲利 / 平均虧損
        - Win Rate: 獲利交易比例
        - Maximum Drawdown: 指定標的市場資料中的最大回撤
        - ROI: 總盈虧 / 總成本
        其中，net_profit 與 total_cost 以 history_positions 中每筆交易計算：
        net_profit = pnl, total_cost = entry_price * size

        Returns:
        dict: 形如 {symbol1: {指標...}, symbol2: {指標...}, ...}
        """
        if self.history_positions.empty:
            return {}
        
        # 依標的分組計算指標
        results = {}
        for symbol in self.data_instance.symbols:
            hp_symbol = self.history_positions[self.history_positions["symbol"] == symbol].copy()
            if hp_symbol.empty:
                continue
            hp_symbol["total_cost"] = hp_symbol["entry_price"].astype(float) * hp_symbol["size"].astype(float)
            hp_symbol["net_profit"] = hp_symbol["pnl"].astype(float)
            
            # 指標計算函式
            def sharp_ratio(net_profit: pd.Series, total_cost: pd.Series, Rf: float) -> float:
                Rp = net_profit / total_cost
                Expectation = Rp.mean()
                variance = ((Rp - Expectation) ** 2).sum() / (Rp.size - 1) if Rp.size > 1 else 0.0
                std = variance ** 0.5 
                sharpe = (Expectation - Rf) / std if std != 0 else float('inf')
                return float(sharpe)
        
            def profit_loss_ratio(net_profit: pd.Series) -> float:
                average_gain = net_profit[net_profit > 0].mean() if net_profit[net_profit > 0].size > 0 else 0.0
                average_loss = abs(net_profit[net_profit < 0].mean()) if net_profit[net_profit < 0].size > 0 else 0.0
                PL_ratio = average_gain / average_loss if average_loss != 0 else float('inf')
                return float(PL_ratio)
        
            def win_rate(net_profit: pd.Series) -> float:
                total_trade_number = net_profit.size
                if total_trade_number == 0:
                    return 0.0
                profit_trade_number = net_profit[net_profit > 0].size
                return float(profit_trade_number / total_trade_number)
        
            def maximum_drawdown(close_series: pd.Series) -> float:
                cumulative_max = close_series.cummax()
                drawdowns = (close_series - cumulative_max) / cumulative_max
                return float(drawdowns.min())
        
            def roi(net_profit: float, total_cost: float) -> float:
                return net_profit / total_cost if total_cost != 0 else 0.0
            
            sharpe = sharp_ratio(hp_symbol["net_profit"], hp_symbol["total_cost"], Rf)
            pl_ratio = profit_loss_ratio(hp_symbol["net_profit"])
            winrate = win_rate(hp_symbol["net_profit"])
            total_net_profit = hp_symbol["net_profit"].sum()
            total_cost = hp_symbol["total_cost"].sum()
            overall_roi = roi(total_net_profit, total_cost)
        
            # 以此標的的市場資料計算最大回撤
            close_series = self.data_instance.original_datas[symbol]["Close"].astype(float)
            mdd = maximum_drawdown(close_series)
        
            results[symbol] = {
                "sharpe_ratio": sharpe,
                "profit_loss_ratio": pl_ratio,
                "win_rate": winrate,
                "maximum_drawdown": mdd,
                "roi": overall_roi
            }
        return results

# End of Spot class
