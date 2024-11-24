import numpy as np
import pandas as pd

class Account(object):
    _instances = {}  

    def __new__(cls, id):
        if id in cls._instances:
            return cls._instances[id]
        else:
            instance = super(Account, cls).__new__(cls)
            cls._instances[id] = instance
            return instance

    def __init__(self, id):
        # 初始化時檢查是否已經初始化過該實例
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        # 帳戶資訊
        self.id = id
        self.asset = 0
        '''
        交易資訊:
        commission: 交易手續費
        orders_book: all order information writing in main.py
        position: False: 未持有頭寸 True:持有頭寸
        '''
        self.commission = 0
        self.orders_book=[]
        self.position = False # False: 未持有頭寸 True:持有頭寸

    def set_asset(self, cash):
        self.asset = cash

    def get_asset(self):
        return self.asset

    def set_commission(self, commission):
        self.commission = commission

    def have_position(self):
        for order in self.orders_book:
            if not order['status']:
                self.position = True
                return True
        self.position = False
        return False

    
class order(object):
    def __init__(self) -> None:
        pass