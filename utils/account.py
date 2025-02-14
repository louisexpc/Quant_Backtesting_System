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
        # 避免重複初始化
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True
        """
        [Account Information]
          - id: account identifier
          - spot_balance: 現貨帳戶 USDT 餘額
          - future_balance: 合約帳戶 USDT 餘額
        """
        self.spot_balance = 0
        self.future_balance = 0
        self.id = id

    def set_future_balance(self,balance:float):
        if balance<0:
            raise ValueError("Future balance shoud larger than 0")
        self.future_balance = balance
    
    def set_spot_balance(self,balance:float):
        if balance<0:
            raise ValueError("Spot balance shoud larger than 0")
        self.spot_balance = balance
    