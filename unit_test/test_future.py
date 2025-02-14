# test_future.py
import unittest
import pandas as pd
import tempfile
import os

from utils.future import Future
from utils.account import Account
from utils.data import DataSingleton

class TestFutureModule(unittest.TestCase):
    def setUp(self):
        # 重置 Singleton
        DataSingleton._instance = None

        # 建立固定測試市場數據
        self.test_data = pd.DataFrame({
            "Datetime": ["2023-01-01 00:00:00", "2023-01-01 04:00:00"],
            "Open": [95000, 96000],
            "High": [95500, 96500],
            "Low": [94500, 95500],
            "Close": [95000, 96000],
            "Volume": [1000, 2000]
        })

        # 存入暫時 CSV 檔案
        self.temp_dir = tempfile.gettempdir()
        self.csv_path = os.path.join(self.temp_dir, "test_market_data.csv")
        self.test_data.to_csv(self.csv_path, index=False)
        self.symbols = ["BTCUSDT"]
        self.data_paths = [self.csv_path]
        self.ds = DataSingleton(self.data_paths, self.symbols)

        # 建立帳戶，初始 future_balance 為 10000 USDT
        self.account = Account("unit_test_account")
        self.account.future_balance = 10000

        # 建立 Future 實例
        self.future = Future(self.account)

        # 為避免隨機滑價影響結果，將 simulate_slippage 替換為恒等函式
        self.future.simulate_slippage = lambda price, slippage_prob=0.0, slippage_range=(-0.005,0.005): price

    def tearDown(self):
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        DataSingleton._instance = None

    def test_insufficient_margin(self):
        """測試若保證金不足，下單應拋出錯誤。"""
        with self.assertRaises(ValueError):
            self.future.create_order(symbol="BTCUSDT", side=1, notional=100000, leverage=5, order_type="market")

    def test_market_order_execution(self):
        """測試市價單下單後，成交數量、保證金與手續費扣除是否正確。"""
        initial_balance = self.account.future_balance
        self.future.create_order(symbol="BTCUSDT", side=1, notional=1000, leverage=5, order_type="market")
        # 預期成交價格為 95000
        execution_price = 95000.0
        expected_quantity = 1000 / execution_price
        required_margin = 1000 / 5  # 200
        expected_fee = 1000 * (0.05 * 0.01)  # 0.5
        expected_balance = initial_balance - (required_margin + expected_fee)
        self.assertEqual(len(self.future.positions), 1)
        pos = self.future.positions.iloc[0]
        self.assertAlmostEqual(float(pos["entry_price"]), execution_price)
        self.assertAlmostEqual(float(pos["size"]), expected_quantity)
        self.assertAlmostEqual(float(pos["margin"]), required_margin)
        self.assertAlmostEqual(self.account.future_balance, expected_balance)

    def test_limit_order_execution(self):
        """測試限價單下單後，當市場價格滿足條件時成交。"""
        initial_balance = self.account.future_balance
        self.future.create_order(symbol="BTCUSDT", side=1, notional=1000, price=94000, leverage=5, order_type="limit")
        self.assertEqual(len(self.future.order_book), 1)
        required_margin = 1000 / 5  # 200
        expected_balance = initial_balance - required_margin
        self.assertAlmostEqual(self.account.future_balance, expected_balance)
        # 模擬市場價格變動，將 Close 設為 93000，觸發多單成交（當前價格 <= 限價單價格）
        self.test_data.loc[0, "Close"] = 93000
        self.ds.original_datas["BTCUSDT"] = self.test_data
        self.future.update_data(self.ds.get_current_index())
        self.assertEqual(len(self.future.order_book), 0)
        self.assertEqual(len(self.future.positions), 1)

    def test_unrealized_pnl_update(self):
        """測試未實現盈虧隨市場價格更新是否正確計算。"""
        self.future.create_order(symbol="BTCUSDT", side=1, notional=1000, leverage=5, order_type="market")
        pos = self.future.positions.iloc[0]
        entry_price = float(pos["entry_price"])
        quantity = float(pos["size"])
        # 模擬市場價格上漲：將 Close 由 95000 變為 97000
        self.test_data.loc[0, "Close"] = 97000
        self.ds.original_datas["BTCUSDT"] = self.test_data
        self.future.update_data(self.ds.get_current_index())
        expected_upnl = (97000 - entry_price) * quantity * 1
        self.assertAlmostEqual(float(self.future.positions.iloc[0]["unrealized_pnl"]), expected_upnl)

    def test_close_position(self):
        """測試平倉時盈虧計算與保證金返還是否正確。"""
        initial_balance = self.account.future_balance
        self.future.create_order(symbol="BTCUSDT", side=1, notional=1000, leverage=5, order_type="market", stop_loss=94000)
        pos = self.future.positions.iloc[0]
        entry_price = float(pos["entry_price"])
        quantity = float(pos["size"])
        margin = float(pos["margin"])
        # 模擬市場價格下跌，觸發止損：將 Close 設為 93000
        self.test_data.loc[0, "Close"] = 93000
        self.ds.original_datas["BTCUSDT"] = self.test_data
        self.future.update_data(self.ds.get_current_index())
        self.assertEqual(len(self.future.positions), 0)
        self.assertGreater(len(self.future.history_positions), 0)
        closed = self.future.history_positions.iloc[-1]
        exit_price = float(closed["exit_price"])
        pnl = (exit_price - entry_price) * quantity
        fee = abs(pnl) * (0.05 * 0.01)
        net_pnl = pnl - fee
        # 帳戶最終餘額應返還 margin + net_pnl
        expected_balance = initial_balance - (1000/5 + 0.5) + (margin + net_pnl)
        self.assertAlmostEqual(self.account.future_balance, expected_balance)

if __name__ == '__main__':
    unittest.main()
