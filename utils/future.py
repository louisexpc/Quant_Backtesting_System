import numpy as np
import pandas as pd

from utils.account import Account
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
class future:

    def __init__(self,data,account:Account):
        """
        Parametes
        Container the information of future
        Order={
            "order_id":訂單 ID
            "symbol":貨幣對
            "entry_price":int,買入價格
            "size":float,買入倉位大小(U本位)
            "timestamp":pd.Datetime,建倉時間
            "side":int, -1:做空 ; 1:做多
            "leverage":int,槓桿大小
            "liquidation_price":強制平倉價格
            ""
        }
        history_positions: 歷史持倉
        postions: 當前持倉
        _order_id: 訂單id
        _idx: price id
        account: 帳戶資訊(Account Object, from utils/account.py)
        """
        self.history_positions = pd.DataFrame()
        self.positions = pd.DataFrame()
        self.order_info=["order_id","symbol","entry_price","size","timestamp","side","leverage"]
        self._order_id = 0
        self._idx = 0
        self.account = account
        pass

    def calculate_liquidation_price(self,current_price:float ,unrealized_pnl:float, other_margin:float, position_size:float):
        """
        全倉模式:
        U本位合約強平價格
        

        Parameters:
        
        unrealized_pnl (float): 当前合约的未实现盈亏（UPNL1）
        other_margin (float): 其他合约的维持保证金总和（TMM1）
        position_size (float): 当前合约的持仓数量

        Returns:
        float: liquidation_price 強制平倉價格
        """
        # 計算當前合約的維持保證金 MM
        mm = position_size * current_price * Maintenance_Margin_Rate

        # 計算 Equity（帳戶淨值）
        equity = self.account.future_balance + unrealized_pnl

        # 計算 TMM1（其他合約的維持保證金總和）
        tmm1 = self.account.total_margins - mm

        # 計算強制平倉價格
        liquidation_price = (equity - tmm1 - mm) / position_size
        return liquidation_price

    def create_order(self,symbol:str, side:int, amount:float, price:float, params):
        liquidation_price = self.calculate_liquidation_price()
        order_data = [[_idx],]
        