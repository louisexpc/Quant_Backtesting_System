import numpy as np
import pandas as pd
import random

from utils.account import Account
from utils.data import DataSingleton
TAKER_COMMISSION = 0.05 * 0.01
MAKER_COMMISSION = 0.02 * 0.01
'''每筆訂單的最低名義價值必須大於 5 USDT 的門檻，實際會根據規定不同'''
'''U本位交易規則參考:https://www.binance.com/en/futures/trading-rules/perpetual'''
MINIMUM_ORDER = 5
Liquidation_Clearance_Fee = 1.25 * 0.01

'''U本位槓桿與保證金:https://www.binance.com/zh-TC/futures/trading-rules/perpetual/leverage-margin'''
'''
初始保證金 = 倉位名義價值 / 開倉槓桿
維持保證金 = 倉位名義價值 * 維持保證金率 - 維持保證金速算額
'''
MAX_LEVERAGE = 125
Maintenance_Margin_Rate = 0.4 * 0.01
'''
清算價格計算:https://www.binance.com/zh-CN/support/faq/%E5%A6%82%E4%BD%95%E8%A8%88%E7%AE%97u%E6%9C%AC%E4%BD%8D%E5%90%88%E7%B4%84%E7%9A%84%E5%BC%B7%E5%B9%B3%E5%83%B9%E6%A0%BC-b3c689c1f50a44cabb3a84e663b81d93
https://www.binance.com/zh-CN/support/faq/%E6%95%B0%E5%AD%97%E8%B4%A7%E5%B8%81%E8%A1%8D%E7%94%9F%E5%93%81?c=4&navId=4#18-36
https://www.binance.com/zh-TC/support/faq/u-%E6%9C%AC%E4%BD%8D%E5%90%88%E7%B4%84%E7%9A%84%E6%A7%93%E6%A1%BF%E5%92%8C%E4%BF%9D%E8%AD%89%E9%87%91-360033162192
'''
class Future:
    """
    U-本位合約交易模擬系統 (Futures Trading Simulation).
    
    Attributes:
      - history_positions: 平倉紀錄 (DataFrame)
      - positions: 當前持倉 (DataFrame)
      - order_book: 待成交限價單 (DataFrame)
      - _order_id: 訂單流水號
      - account: 帳戶資訊 (Account 物件)
      - data_instance: DataSingleton 取得市場數據
      - current_data: 最新市場數據 snapshot
    """
    def __init__(self, account: Account):
        """
        Initialize Future trading simulation.
        訂單格式包含：
          - order_id: 訂單 ID
          - symbol: 交易標的
          - entry_price: 建倉價格
          - size: 實際成交的數量（標的資產數量）
          - timestamp: 建倉時的市場索引
          - side: 1 表示多單, -1 表示空單
          - leverage: 槓桿倍數
          - liquidation_price: 強制平倉價格
          - take_profit: (可選) 限價止盈價格
          - stop_loss: (可選) 限價止損價格
          - margin: 保證金
        """
        self.history_positions = pd.DataFrame()
        self.positions = pd.DataFrame()
        self.order_info = ["order_id", "symbol", "entry_price", "size", "timestamp", "side", "leverage", "margin"]
        self.order_book = pd.DataFrame(
            columns=["order_id", "symbol", "price", "size", "side", "status", "leverage", "take_profit", "stop_loss", "margin"]
        )
        self._order_id = 0
        self.account = account
        # 取得 DataSingleton 物件（若上層已初始化，這裡會取得先前資料）
        self.data_instance = DataSingleton([], [])
        #print(f"initial future data instance:\n{self.data_instance.get_current_index()}\n{self.data_instance.get_current_data()}")
        # 註冊 callback 更新市場資料與訂單檢查
        self.data_instance.register_callback(self.update_data)
        # 初始化時取得市場資料，避免 self.current_data 為空
        self.current_data = self.data_instance.get_current_data()

    def update_data(self, idx):
        """
        當 DataSingleton 更新索引時觸發此 callback，
        更新市場數據、檢查掛單、重新計算未實現盈虧，並檢查止盈／止損條件。
        """
        self.current_data = self.data_instance.get_current_data()
        self.execute_order()           # 檢查掛單是否符合成交條件
        self.calculate_unrealized_pnl()  # 更新未實現盈虧
        self.close_position()          # 檢查是否有部位觸發清算或 TP/SL 條件，自動平倉

    def calculate_fees(self, order_value: float, order_type: str = "taker") -> float:
        """計算交易手續費"""
        fee_rate = TAKER_COMMISSION if order_type == "taker" else MAKER_COMMISSION
        return order_value * fee_rate

    def calculate_unrealized_pnl(self):
        """計算未實現盈虧 (UPNL) 並更新持倉資料"""
        if self.positions.empty:
            return
        
        

        def compute_upnl(row: pd.Series):
            symbol = row["symbol"]
            if symbol in self.current_data and self.current_data[symbol] is not None:
                current_price = self.current_data[symbol]["Close"]
                
                return (current_price - row["entry_price"]) * row["size"] * row["side"]
            return 0.0

        self.positions["unrealized_pnl"] = self.positions.apply(compute_upnl, axis=1)
        #print(f"Positions:\n{self.positions}")

    def calculate_liquidation_price(self, symbol: str, quantity: float, entry_price: float, side: int, leverage: int) -> float:
        """
        計算強制平倉價格 (Liquidation Price)。
        此處採用簡化公式：
          - 多單 (side=1): liquidation_price = entry_price * (1 - 1/leverage + MAINTENANCE_MARGIN_RATE)
          - 空單 (side=-1): liquidation_price = entry_price * (1 + 1/leverage - MAINTENANCE_MARGIN_RATE)
        """
        if symbol not in self.current_data or self.current_data[symbol] is None:
            raise ValueError(f"Market data for {symbol} not available.")
        if side == 1:
            liquidation_price = entry_price * (1 - 1/leverage + Maintenance_Margin_Rate)
        elif side == -1:
            liquidation_price = entry_price * (1 + 1/leverage - Maintenance_Margin_Rate)
        else:
            raise ValueError("Invalid side value. Must be 1 (long) or -1 (short).")
        return liquidation_price

    def create_order(self, symbol: str, side: int, notional: float, price: float = None,
                     leverage: int = 5, order_type="market",
                     take_profit: float = None, stop_loss: float = None):
        """
        創建訂單，支援市價單與限價單，並可附帶限價止盈與止損。
        此處 notional 表示訂單總名義價值（以 USDT 計）。
        
        下單前會檢查帳戶餘額是否足以支付所需的保證金，
        保證金 = notional / leverage。
        
        Parameters:
          - symbol: 交易幣種
          - side: 1 (多單) 或 -1 (空單)
          - notional: 訂單總名義價值 (USDT)
          - price: 限價單時必填 (USDT)
          - leverage: 槓桿倍數
          - order_type: "market" 或 "limit"
          - take_profit: (可選) 限價止盈價格 (USDT)
          - stop_loss: (可選) 限價止損價格 (USDT)
        """
        if symbol not in self.current_data or self.current_data[symbol] is None:
            raise ValueError(f"Market data for {symbol} not available.")

        required_margin = notional / leverage
        if self.account.future_balance < required_margin:
            #raise ValueError("Insufficient funds to open position (margin requirement not met).")
            print(f"Insufficient funds to open position (margin requirement not met).")
            return
        
        # 市價單：使用當前成交價計算實際數量
        if order_type == "market":
            execution_price = self.simulate_slippage(self.current_data[symbol]["Close"])
            quantity = notional / execution_price
            # 市價單下單時，立即扣除保證金
            self.account.future_balance -= required_margin
            self._execute_order(symbol, side, quantity, execution_price, leverage, order_type,
                                take_profit, stop_loss, reserved_margin=required_margin)
        elif order_type == "limit":
            if price is None:
                raise ValueError("Price must be provided for limit orders.")
            # 限價單：用設定的限價計算數量
            quantity = notional / price
            # 限價單建立時，預先扣除保證金
            self.account.future_balance -= required_margin
            order_record = {
                "order_id": self._order_id,
                "symbol": symbol,
                "price": price,
                "size": quantity,
                "side": side,
                "status": "pending",
                "leverage": leverage,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "margin": required_margin
            }
            self.order_book = pd.concat([self.order_book, pd.DataFrame([order_record])], ignore_index=True)
            self._order_id += 1
        else:
            raise ValueError(f"Unsupported order type: {order_type}")

    def _execute_order(self, symbol: str, side: int, quantity: float, execution_price: float,
                       leverage: int, order_type: str, take_profit: float = None, stop_loss: float = None,
                       reserved_margin: float = None):
        """
        內部方法：執行訂單，檢查最低訂單價值、計算手續費、計算清算價格，
        並建立部位，同時記錄已預留的保證金。
        """
        order_value = execution_price * quantity
        if order_value < MINIMUM_ORDER:
            #raise ValueError(f"Order value must be at least {MINIMUM_ORDER} USDT.")
            print(f"Insufficient funds to open position (margin requirement not met).")
            return
        fee_type = "taker" if order_type == "market" else "maker"
        fees = self.calculate_fees(order_value, fee_type)
        # 若 reserved_margin 已提供，則使用之；否則，扣除保證金（理論上應該總是有 reserved_margin）
        if reserved_margin is None:
            required_margin = order_value / leverage
            if self.account.future_balance < required_margin:
                raise ValueError("Insufficient funds to open position (margin requirement not met).")
            self.account.future_balance -= required_margin
            margin = required_margin
        else:
            margin = reserved_margin
        liquidation_price = self.calculate_liquidation_price(symbol, quantity, execution_price, side, leverage)
        order = {
            "order_id": self._order_id,
            "symbol": symbol,
            "entry_price": execution_price,
            "size": quantity,
            "timestamp": self.data_instance.get_current_index(),
            "datetime":self.data_instance.get_current_data()[symbol]["Datetime"],
            "side": side,
            "leverage": leverage,
            "liquidation_price": liquidation_price,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "margin": margin
        }
        print(f"order execute:\n{order}")
        self._order_id += 1
        # 扣除手續費
        self.account.future_balance -= fees
        self.positions = pd.concat([self.positions, pd.DataFrame([order])], ignore_index=True)

    def execute_order(self):
        """
        處理 order_book 中所有待成交的限價單：
          - 對多單：當前價格 <= 限價單價格時成交；
          - 對空單：當前價格 >= 限價單價格時成交。
        成交後呼叫 _execute_order 並傳入預留保證金，然後從掛單中移除該訂單。
        """
        if self.order_book.empty:
            return
        
        def check_and_execute(order:pd.Series):
            """
            apply() 的回傳值會是一個 與 DataFrame 列數相同的 Series，每個元素是 check_and_execute(row) 的回傳結果。
            """
            symbol = order["symbol"]
            if symbol not in self.current_data or self.current_data[symbol] is None:
                return False
            current_price = self.current_data[symbol]['Close']
            if (order["side"] == 1 and current_price <= order["price"]) or \
               (order["side"] == -1 and current_price >= order["price"]):
                reserved_margin = order.get("margin")
                self._execute_order(symbol, order["side"], order["size"], order["price"],
                                    order["leverage"], order_type="maker",
                                    take_profit=order.get("take_profit"),
                                    stop_loss=order.get("stop_loss"),
                                    reserved_margin=reserved_margin)
                return True
            return False
        #executed_orders 會是一個 Series，其索引與 self.order_book 相同，每個元素的值是 True 或 False。
        executed_orders = self.order_book.apply(check_and_execute, axis=1)
        #self.order_book[executed_orders].index 中.index 會取出這些行的索引。
        #self.order_book[executed_orders].index
        #會等於: Int64Index([0, 2], dtype='int64')
        self.order_book.drop(self.order_book[executed_orders].index, inplace=True)

    def simulate_slippage(self, price, slippage_prob=0.001, slippage_range=(-0.005, 0.005)):
        """模擬滑價效應"""
        if random.random() < slippage_prob:
            slippage = random.uniform(*slippage_range)
            return price * (1 + slippage)
        return price

    def close_position(self, order_id: int = None):
        """
        平倉作業：
          - 若傳入 order_id，則手動平倉該部位；
          - 否則，自動檢查所有持倉，若觸發清算或止盈／止損條件則平倉。
          
        自動平倉條件：
          對於多單 (side=1)：
            - 清算條件：當前價格 <= liquidation_price
            - 止盈條件：若設定 take_profit 且當前價格 >= take_profit
            - 止損條件：若設定 stop_loss 且當前價格 <= stop_loss
          對於空單 (side=-1)：
            - 清算條件：當前價格 >= liquidation_price
            - 止盈條件：若設定 take_profit 且當前價格 <= take_profit
            - 止損條件：若設定 stop_loss 且當前價格 >= stop_loss
          
        平倉後計算盈虧、扣除手續費，將部位移至 history_positions，並將原先預留的保證金加上盈虧回補到帳戶餘額中。
        Returns:
          平倉部位的 DataFrame
        """
        if self.positions.empty:
            return None
        
        #指定平倉
        if order_id is not None:
            position = self.positions[self.positions["order_id"] == order_id]
            if position.empty:
                return None
        else:
            def close_condition(row:pd.Series):
                symbol = row["symbol"]
                if symbol not in self.current_data or self.current_data[symbol] is None:
                    return False
                current_price = self.current_data[symbol]["Close"]
                if row["side"] == 1:
                    liquidation_triggered = current_price <= row["liquidation_price"]
                    tp_triggered = (pd.notnull(row.get("take_profit")) and current_price >= row["take_profit"])
                    sl_triggered = (pd.notnull(row.get("stop_loss")) and current_price <= row["stop_loss"])
                elif row["side"] == -1:
                    liquidation_triggered = current_price >= row["liquidation_price"]
                    tp_triggered = (pd.notnull(row.get("take_profit")) and current_price <= row["take_profit"])
                    sl_triggered = (pd.notnull(row.get("stop_loss")) and current_price >= row["stop_loss"])
                else:
                    return False
                return liquidation_triggered or tp_triggered or sl_triggered

            position = self.positions[self.positions.apply(close_condition, axis=1)]
        if position.empty:
            return None

        closed_positions = []
        for _, pos in position.iterrows():
            symbol = pos["symbol"]
            if symbol not in self.current_data or self.current_data[symbol] is None:
                continue
            # 計算收益與手續費
            exit_price = self.simulate_slippage(self.current_data[symbol]["Close"])
            fees = self.calculate_fees(exit_price * pos["size"])
            pnl = (exit_price - pos["entry_price"]) * pos["size"] * pos["side"] - fees
            
            # 將原先預留的 margin 回補，加上盈虧後更新帳戶餘額
            margin = pos.get("margin", 0)
            self.account.future_balance += (margin + pnl)
            closed_position = pos.to_dict()
            closed_position["exit_price"] = exit_price
            closed_position["exit_timestamp"] = self.data_instance.get_current_index()
            closed_position["exit_datetime"] = self.data_instance.get_current_data()[symbol]["Datetime"]
            closed_position["pnl"] = pnl
            closed_positions.append(closed_position)
        self.positions = self.positions.drop(position.index)
        self.history_positions = pd.concat([self.history_positions, pd.DataFrame(closed_positions)], ignore_index=True)
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

if __name__ =="__main__":
    symbols = ["BTCUSDT",""]
    data_paths = [r"C:\Users\louislin\OneDrive\桌面\data_analysis\backtesting_system\test.csv"]
    
    ds = DataSingleton(data_paths, symbols)
    print("DataSingleton 初始化完成，載入市場數據：")
    print(ds.original_datas)
    account = Account("test_account")
    account.set_future_balance(10000)  # 設定 10000 USDT

    # --- Step 4: 建立 Future 實例 ---
    future = Future(account)
    print(ds.get_current_data())
    future.create_order(symbols[0],1,1000,price=1000,order_type="limit",take_profit=1200)
    print(ds.get_current_data())
    print(f"下單後帳戶餘額變化:{account.future_balance}")
    print(f"{future.order_book}")
    ds.run()
    # print(f"訂單觸發:{ds.get_current_data()}")
    # print(f"訂單執行帳戶餘額變化:{account.future_balance}")
    # print(f"{future.order_book}")
    # print(f"{future.positions}")
    ds.run()
    print(f"平倉:{ds.get_current_data()}")
    print(f"平倉帳戶餘額變化:{account.future_balance}")
    print(f"{future.order_book}")
    print(f"{future.positions}")
    print(f"{future.history_positions}")

    

    pass