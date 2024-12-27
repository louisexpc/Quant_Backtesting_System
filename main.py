import pandas as pd

from utils.account import Account
from utils.order import order_history,order
from utils.evalution import roi,maximum_drawdown
from pkg.ConfigLoader import config
from strategy.stoch_rsi import StochRSI
from strategy.macd_cross import macd_cross

#STRATEGY_CONFIG = ".\\strategy\\stoch_rsi.json"
#STRATEGY_NAME = "stoch_rsi"
STRATEGY_CONFIG = r".\strategy\macd_cross.json"
STRATEGY_NAME = "macd_cross"
class backtest(object):
    def __init__(self,account_id:int,data_list:list,symbols:list):
        self.data_list = data_list
        self.symbols = symbols
        self.orginal_data = self.trans_original_data()

        self.account =Account(account_id)
        self.history = self.account.history
        self.current_index = 0
        self.risk_management = 0.05 #基於1%原則，每一position 的失效點(unit:%)
        """ Init Sotch RSI Strategy """
        """
        self.config = config(STRATEGY_CONFIG).load_config()[STRATEGY_NAME]
        self.timeframe = self.config['timeframe']
        self.limit = self.config['limit']
        self.param = self.config['param']
        """
        """ Init MACD Cross Strategy """
        self.config = config(STRATEGY_CONFIG).load_config()[STRATEGY_NAME]
        self.limit = self.config['limit']
        self.param = self.config['param']

    def trans_original_data(self)->dict:
        original_data = {}
        for i in range(len(self.data_list)):
            original_data[self.symbols[i]]=pd.read_csv(self.data_list[i])
        return original_data
    
    def data_segment(self)->dict:
        i = self.current_index
        segment_data = {}
        for symbol in self.symbols:
            if i>self.limit:
                df = self.orginal_data[symbol].iloc[i-self.limit:i,:]
                segment_data[symbol]=df
            else:
                df = self.orginal_data[symbol].iloc[0:i,:]
                segment_data[symbol]=df
        #print("test segment_data:\n",segment_data)
        return segment_data
            
    def next(self):
        i = self.current_index
        
        if i>=self.limit:
            current_data = self.data_segment()
            #signals = StochRSI(current_data).run()
            signals = macd_cross(current_data).run()
            #print(f"{i}:{signals}")
            for index in range(len(signals)):
                if signals[self.symbols[index]]==1:
                    self.buy(self.symbols[index])
                if self.history.have_positions():
                    orders = self.history.search_positions()
                    if signals[self.symbols[index]]==-1:
                        for index,order in orders.iterrows():
                            self.sell(order)


        

    def run(self):
        # Iteration whole datasets
        while self.current_index < len(self.orginal_data[self.symbols[0]]):
            self.next()
            self.current_index += 1
        # 計算持倉價值
        position_value = 0
        if self.history.have_positions():
            positions = self.history.search_positions()
            for order in positions.itertuples():
                price = self.orginal_data[order.symbol]['Close'].iloc[-1]
                position_value+=price*order.amount

        print(f"Testing Down\nAccount ID: {self.account.id}\nFinal asset: {self.account.asset}, position value: {position_value} Total:{self.account.asset+position_value}")
        self.history.evaluation()
    
    def calculate_position(self):
        #頭寸大小 = 賬戶大小*賬戶風險(1%)/失效點
        position = (self.account.asset * 0.01) /self.risk_management
        return position
    
    #create(buy) a single order
    def buy(self,symbol:str):
        current_price = self.orginal_data[symbol]['Close'].iloc[self.current_index]
        # 計算總花費，包括手續費
        total_cost = self.calculate_position()
        #最小下單金額(based on binance API 5 USDT)
        if total_cost<5: 
            print("Buy order failed: Your account balance is insufficient")
            return
        # 計算手續費
        commission_paid = total_cost * self.account.commission
        # 計算實際用於購買資產的金額
        net_spend = total_cost - commission_paid
        # 計算購買的資產數量
        amount = net_spend / current_price
        print(f"{symbol},{self.orginal_data[symbol]['Datetime'][self.current_index]} price:{current_price:.4f} buy amount:{amount:.4f},spend:{net_spend:.4f}")
        '''
        更新帳戶資訊
        '''
        self.account.asset -= total_cost
        order_info = {
            'symbol':symbol,
            'Datetime': self.orginal_data[symbol]['Datetime'].iloc[self.current_index],
            'id':self.current_index,
            'buy_price': current_price,
            'amount': amount,
            'type': 'buy',
            'commission': commission_paid,
            'total_cost':total_cost,
            'status':False, #False: 尚未賣出 , True: 賣出
            'sell_time':"",
            "sell_price":0.0,
            'net_gain':0.0,
            'net_profit':0.0
        }
        self.history.update_history(order(order_info))
       

    #sell single order
    def sell(self,order:pd.Series):

        if not order['status']:
            current_price = self.orginal_data[order['symbol']]['Close'].iloc[self.current_index]
            amount = order['amount']
            # 計算總收益
            gross_value = current_price * amount
            # 計算手續費
            commission_paid = gross_value * self.account.commission
            # 計算淨獲利
            net_gain = gross_value - commission_paid
            '''
            更新帳戶資訊
            '''
            self.account.asset += net_gain
            update_info ={
                'sell_time':self.orginal_data[order['symbol']]['Datetime'].iloc[self.current_index],
                'sell_price':current_price,
                'net_gain':net_gain,
                'status':True,
                'net_profit':net_gain-order['total_cost']
            }
            self.history.update_order(order['id'],update_info) 
            #compute ROI
            net_profit = float(net_gain-order['total_cost'])
            r = roi( net_profit,order['total_cost'])
            #compute MDD 最大回撤
            MDD = 100*maximum_drawdown(order['id'],self.current_index,self.orginal_data[order['symbol']]['Close'])
            print(f"{order['symbol']},{order['sell_time']} price: {current_price:.4f} sell amount: {amount:.4f}, net gain: {net_gain:.4f}, net profit:{net_profit:.3f}, ROI:{r:.3f}, MDD:{MDD:.2f}%")
            
    
            

if __name__=='__main__':
    timeframe = '1d'
    data_list=[]
    symbols = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT"]
    for symbol in symbols:
        data_list.append(rf".\data\{timeframe}_{symbol}.csv")
 
    trader = backtest(0,data_list,symbols)
    trader.account.set_asset(1000)
    trader.account.set_commission(0.001)
    trader.run()