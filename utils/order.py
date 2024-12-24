import pandas as pd

from utils.evalution import profit_loss_ratio,sharp_ratio,win_rate

class order(object):
    def __init__(self, info: dict) -> None:
        '''
        info 應該是一個字典，包含訂單的相關信息
        '''
        self.order_info = pd.DataFrame([info])  # 將字典包裝成列表後傳入 DataFrame

class order_history(object):
    def __init__(self):
        self.history = pd.DataFrame()
    
    def update_history(self, new_order: order) -> None:
        '''
        合併新 order 至 history
        '''
        self.history = pd.concat([self.history, new_order.order_info], axis=0, ignore_index=True)
    
    def search_order(self, id: int) -> pd.DataFrame:
        '''
        給定 id 搜尋目標 order
        '''
        return self.history[self.history['id'] == id]
    
    def update_order(self, id: int, update_info: dict) -> None:
        '''
        給定 id 更新指定 order
        '''
        for key, info in update_info.items():
            #self.history[self.history['id'] == id][key] = info
            # print(f"test:{self.history.loc[self.history['id'] == id, key]}")
            self.history.loc[self.history['id'] == id, key] = info
    def search_positions(self)->pd.DataFrame:
        '''
        回傳 history 中尚未平倉的 orders
        '''
        if not self.have_positions():
            return None
        return self.history[self.history['status']==False]
    
    def have_positions(self)->bool:
        '''
        查詢 history 中是否有尚未平倉的 orders
        '''
        if self.history.empty:
            return False
        return not self.history[self.history['status']==False].empty
    
    def evaluation(self)->None:
        '''
        評估策略表現，計算 sharp ratio , PLR, win rate
        '''
        print(self.history)
        net_profit = self.history['net_profit']
        total_cost = self.history['total_cost']
        sharpRatio = sharp_ratio(net_profit,total_cost,0.008)
        plr = profit_loss_ratio(net_profit)
        winRate = win_rate(net_profit)
        print(f"sharp ratio: {sharpRatio:.3f}\nProfit-Loss Ratio: {plr:.3f}\nWin Rate:{winRate:.3f}")
    
# 測試程式碼
if __name__ == '__main__':
    # 測試初始化 order
    info1 = {
        'symbol': 'BTC/USDT',
        'Datetime': '2024-12-24',
        'id': 15,
        'buy_price': 155.2,
        'amount': 155.66,
        'type': 'buy',
        'commission': 1,
        'total_cost': 200,
        'status': False,  # False: 尚未賣出 , True: 賣出
        'sell_time': "",
        "sell_price": 0,
        'net_profit': 0
    }
    info2 = {
        'symbol': 'ETH/USDT',
        'Datetime': '2024-12-24',
        'id': 16,
        'buy_price': 50.2,
        'amount': 100.5,
        'type': 'buy',
        'commission': 0.5,
        'total_cost': 150,
        'status': False,
        'sell_time': "",
        "sell_price": 0,
        'net_profit': 0
    }
    order1 = order(info1)
    order2 = order(info2)
    
    # 測試 order_history
    history = order_history()
    history.update_history(order1)
    history.update_history(order2)
    
    # 顯示歷史訂單
    print("歷史訂單：")
    print(history.history)
    
    # 搜尋特定訂單
    print("\n搜尋 id 為 15 的訂單：")
    print(history.search_order(15))
    #搜索未售出訂單
    print("\n未售出訂單：")
    print(type(history.search_positions()))
    print(history.search_positions())
    # 更新訂單資訊
    update_data = {'sell_price': 160, 'status': True, 'sell_time': '2024-12-25'}
    history.update_order(15, update_data)
    
    # 顯示更新後的歷史訂單
    print("\n更新後的歷史訂單：")
    print(history.history)

    #搜索未售出訂單
    print("\n未售出訂單：")
    print(type(history.search_positions()))
    print(history.search_positions())

    update_data = {'sell_price': 150, 'status': True, 'sell_time': '2024-12-25'}
    history.update_order(16, update_data)
    # 顯示更新後的歷史訂單
    print("\n更新後的歷史訂單：")
    print(history.history)


    #搜索未售出訂單
    print("\n未售出訂單：")
    print(type(history.search_positions()))
    print(history.search_positions())
    print(history.have_positions())

    #type
    history.evaluation()
