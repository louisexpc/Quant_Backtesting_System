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
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True
        '''
        Account Information]
        id: account id
        spot_balance:現貨帳戶 USDT 餘額
        future_balance:合約帳戶 USDT 餘額

        '''
        self.spot_balance = 0
        self.future_balance = 0
        self.id = id
        self.asset = 0
        '''
        交易資訊:
        commission: 交易手續費
        history: order_history 物件，紀錄訂單/持倉
        '''
        self.commission = 0
        self.history=order_history()
        '''
        future:
        total_margins: float, 所有合約保證金總和
        '''
        self.total_margins = 0
        

    def set_asset(self, cash):
        self.asset = cash

    def get_asset(self):
        return self.asset

    def set_commission(self, commission):
        self.commission = commission

    

