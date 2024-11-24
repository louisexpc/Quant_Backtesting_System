import json

class config(object):
    # class variable: all instance share the variable to check config
    _instance = {}

    # 在 __init__前先呼叫，確保所有config路徑創建的instance只有一個(singleton pattern)，cls表示參數本身
    def __new__(cls,config_path):
        if config_path not in cls._instance:
            '''
            super(config,cls):從 Config 的父類(即object)中尋找方法，並且我要傳遞 cls 這個類本身作為參數。
            .__new__(cls):cls 參數是傳遞給父類 __new__ 方法的。cls 代表的是要創建的類別，也就是當前的 Config 類別。
            '''
            cls._instance[config_path]=super(config,cls).__new__(cls)
            cls._instance[config_path].config_file_path = config_path
            cls._instance[config_path].config = None
        
        return cls._instance[config_path]
    
    def load_config(self):
        if self.config ==None:
            with open(self.config_file_path,"r") as f:
                self.config = json.load(f)
        return self.config
    
