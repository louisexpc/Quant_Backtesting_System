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
