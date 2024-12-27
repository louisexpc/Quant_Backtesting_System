from utils.order import order_history
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
        history: order_history 物件，紀錄訂單/持倉
        '''
        self.commission = 0
        self.history=order_history()
        

    def set_asset(self, cash):
        self.asset = cash

    def get_asset(self):
        return self.asset

    def set_commission(self, commission):
        self.commission = commission

    

