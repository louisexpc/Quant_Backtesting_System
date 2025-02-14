import unittest
import pandas as pd
import tempfile
import os

from utils.spot import Spot
from utils.account import Account
from utils.data import DataSingleton

class TestSpotModule(unittest.TestCase):
    def setUp(self):
        # 重置 DataSingleton
        DataSingleton._instance = None

        # 建立固定市場數據
        self.test_data = pd.DataFrame({
            "Datetime": ["2023-01-01 00:00:00", "2023-01-01 04:00:00"],
            "Open": [95000, 96000],
            "High": [95500, 96500],
            "Low": [94500, 95500],
            "Close": [95000, 96000],
            "Volume": [1000, 2000]
        })

        # 儲存 CSV
        self.temp_dir = tempfile.gettempdir()
        self.csv_path = os.path.join(self.temp_dir, "test_spot_data.csv")
        self.test_data.to_csv(self.csv_path, index=False)
        self.symbols = ["BTCUSDT"]
        self.data_paths = [self.csv_path]
        self.ds = DataSingleton(self.data_paths, self.symbols)

        # 建立帳戶，初始現金餘額 10000 USDT
        self.account = Account("spot_test_account")
        self.account.spot_balance = 10000

        # 建立 Spot 模組實例
        self.spot = Spot(self.account)
        # 將 simulate_slippage 設為恆等函式以穩定測試結果
        self.spot.simulate_slippage = lambda price, slippage_prob=0.0, slippage_range=(-0.005,0.005): price

    def tearDown(self):
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        DataSingleton._instance = None

    def test_insufficient_funds(self):
        """測試若現金不足，下單應拋出錯誤。"""
        with self.assertRaises(ValueError):
            self.spot.create_order(symbol="BTCUSDT", notional=20000, order_type="market")

    def test_market_order_execution(self):
        """測試市價單下單成交後，數量與扣款是否正確更新。"""
        initial_balance = self.account.spot_balance
        self.spot.create_order(symbol="BTCUSDT", notional=1000, order_type="market")
        execution_price = 95000.0  # 預期成交價格
        fees = 1000 * 0.001  # 手續費 = 1 USDT
        net_notional = 1000 - fees
        expected_quantity = net_notional / execution_price
        # 市價單下單後，扣款應在 _execute_order 中執行一次，扣除 notional
        expected_balance = initial_balance - 1000
        self.assertEqual(len(self.spot.positions), 1)
        pos = self.spot.positions.iloc[0]
        self.assertAlmostEqual(float(pos["entry_price"]), execution_price)
        self.assertAlmostEqual(float(pos["size"]), expected_quantity)
        self.assertAlmostEqual(self.account.spot_balance, expected_balance)

    def test_limit_order_execution(self):
        """測試限價單掛單後，當市場價格達成交條件時成交。"""
        initial_balance = self.account.spot_balance
        self.spot.create_order(symbol="BTCUSDT", notional=1000, price=94000, order_type="limit")
        self.assertEqual(len(self.spot.order_book), 1)
        expected_balance = initial_balance - 1000
        self.assertAlmostEqual(self.account.spot_balance, expected_balance)
        # 模擬市場價格下降觸發限價單成交：將 Close 設為 93000
        self.test_data.loc[0, "Close"] = 93000
        self.ds.original_datas["BTCUSDT"] = self.test_data
        self.spot.update_data()
        self.assertEqual(len(self.spot.order_book), 0)
        self.assertEqual(len(self.spot.positions), 1)

    def test_unrealized_pnl_update(self):
        """測試未實現盈虧隨市場價格變化是否正確更新。"""
        self.spot.create_order(symbol="BTCUSDT", notional=1000, order_type="market")
        pos = self.spot.positions.iloc[0]
        entry_price = float(pos["entry_price"])
        quantity = float(pos["size"])
        # 模擬市場價格上漲：將 Close 由 95000 改為 97000
        self.test_data.loc[0, "Close"] = 97000
        self.ds.original_datas["BTCUSDT"] = self.test_data
        self.spot.update_data()
        expected_upnl = (97000 - entry_price) * quantity
        self.assertAlmostEqual(float(self.spot.positions.iloc[0]["unrealized_pnl"]), expected_upnl)

    def test_close_position(self):
        """測試平倉時盈虧計算與現金返還是否正確。"""
        initial_balance = self.account.spot_balance
        # 下市價單 1000 USDT，設定止損 94000
        self.spot.create_order(symbol="BTCUSDT", notional=1000, order_type="market", stop_loss=94000)
        pos = self.spot.positions.iloc[0]
        entry_price = float(pos["entry_price"])
        quantity = float(pos["size"])
        print(f"order book:\n{self.spot.order_book}")
        # 模擬市場價格下跌觸發止損：將 Close 設為 93000
        self.test_data.loc[0, "Close"] = 93000
        self.ds.original_datas["BTCUSDT"] = self.test_data
        self.spot.update_data()
        self.assertEqual(len(self.spot.positions), 0)
        self.assertGreater(len(self.spot.history_positions), 0)
        closed = self.spot.history_positions.iloc[-1]
        exit_price = float(closed["exit_price"])
        pnl = (exit_price - entry_price) * quantity
        fee = abs(pnl) * 0.001
        net_pnl = pnl - fee
        expected_balance = initial_balance - 1000 + net_pnl
        self.assertAlmostEqual(self.account.spot_balance, expected_balance)

if __name__ == '__main__':
    unittest.main()
