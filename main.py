from utils.indicator import RSI,StochasticRSI,ExponentialMovingAverage,MACD
from utils.account import Account
import pandas as pd
from pkg.ConfigLoader import config
from strategy.stoch_rsi import StochRSI
STRATEGY_CONFIG = "C:\\Users\\louislin\\OneDrive\\桌面\\data_analysis\\backtesting_system\\strategy\\stoch_rsi.json"
STRATEGY_NAME = "stoch_rsi"
class backtest(object):
    def __init__(self,account_id:int,data_list:list,symbols:list):
        self.data_list = data_list
        self.symbols = symbols
        self.orginal_data = self.trans_original_data()
        self.account =Account(account_id)
        self.current_index = 0
        self.risk_management = 0.05 #基於1%原則，每一position 的失效點(unit:%)
        """ Init Strategy """
        self.config = config(STRATEGY_CONFIG).load_config()[STRATEGY_NAME]
        self.timeframe = self.config['timeframe']
        self.limit = self.config['limit']
        self.param = self.config['param']
        '''
        indicator
        
        self.rsi = RSI(data,'Close',20,30)
        self.short_rsi = self.rsi.get_short_rsi()
        self.long_rsi = self.rsi.get_long_rsi()
        self.stoch_period = 14
        self.ema_peroiod = 3
        self.stochRSI = StochasticRSI(data,'Close',self.stoch_period).get_stochastic_rsi()*100
        self.ema = ExponentialMovingAverage(data,'Close',self.ema_peroiod).get_ema()
        self.macd=MACD(data)
        self.macd_line = self.macd.get_MACD()
        self.signal_line = self.macd.get_signal()
        '''
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
            
        pass
    def next(self):
        i = self.current_index
        
        if i>=1:
            current_data = self.data_segment()
            signals = StochRSI(current_data).run()
            for index in range(len(signals)):
                if signals[self.symbols[index]]==1:
                    self.buy(self.symbols[index])
                if self.account.have_position():
                    if signals[self.symbols[index]]==-1:
                        for order in self.account.orders_book:
                            self.sell(order)
                        pass
                    elif signals[self.symbols[index]]==0:
                        for order in self.account.orders_book:
                            if not order['status'] and order['symbol']==self.symbols[index]:
                                current_price = self.orginal_data[self.symbols[index]]['Close'].iloc[self.current_index]
                                amount = order['amount']
                                # 計算總收益
                                gross_value = current_price * amount
                                # 計算手續費
                                commission_paid = gross_value * self.account.commission
                                # 計算實際獲得的金額
                                net_value = gross_value - commission_paid
                                roi = self.compute_ROI( (net_value-order['cost']),order['cost'])
                                if roi>0.1 :
                                    print("止盈")
                                    self.sell(order)
                                elif roi<-0.05:
                                    print("止損")
                                    self.sell(order)

                        
        

    """ def next(self):
        # 確保有足夠多的資料計算出價格趨勢
        i = self.current_index
        if i>=1:
            # 檢查是否為上升趨勢或下降趨勢
            macd_current = self.macd_line.iloc[i]
            macd_prev = self.macd_line.iloc[i - 1]
            signal_current = self.signal_line.iloc[i]
            signal_prev = self.signal_line.iloc[i - 1]
            # 当 MACD 线从下方向上穿过信号线，且 MACD 线小于 0，视为上升趋势的开始
            is_uptrend = (macd_prev < signal_prev) and (macd_current > signal_current) and (macd_current < 0)
            #is_uptrend = (macd_prev < signal_prev) and (macd_current > signal_current)

            # 当 MACD 线从上方向下穿过信号线，且 MACD 线大于 0，视为下降趋势的开始
            is_downtrend = (macd_prev > signal_prev) and (macd_current < signal_current) and (macd_current > 0)
            #is_downtrend = (macd_prev > signal_prev) and (macd_current < signal_current)
            price_buffer = self.ema.iloc[i] * 0.001  # 0.1% 的缓冲区

            #入場策略:stoch rsi < 20 (超賣) 且價格為下降趨勢 且current ema > current close price
            if is_downtrend and self.stochRSI.iloc[i]<20 and self.close.iloc[i]<=self.ema.iloc[i]:
                self.buy()
            #出場策略:stoch rsi > 80(超買) 且價格為上升趨勢 且current ema < current close price
            if self.account.have_position():
                if is_uptrend and self.stochRSI.iloc[i]>80 and self.close.iloc[i]>=self.ema.iloc[i]:
                    for order in self.account.orders_book:
                        self.sell(order)
                #止盈止損
                for order in self.account.orders_book:
                    if not order['status']:
                        current_price = self.close.iloc[self.current_index]
                        amount = order['amount']
                        # 計算總收益
                        gross_value = current_price * amount
                        # 計算手續費
                        commission_paid = gross_value * self.account.commission
                        # 計算實際獲得的金額
                        net_value = gross_value - commission_paid
                        roi = self.compute_ROI( (net_value-order['cost']),order['cost'])
                        if roi>0.1 :
                            print("止盈")
                            self.sell(order)
                        elif roi<-0.05:
                            print("止損")
                            self.sell(order) """
            

           

        

    def run(self):
        while self.current_index < len(self.orginal_data[self.symbols[0]]):
            self.next()
            self.current_index += 1
        # 計算持倉價值
        position_value = 0
        if self.account.have_position():
            
            for order in self.account.orders_book:
                if not order['status']:
                    price = self.orginal_data[order['symbol']]['Close'].iloc[-1]
                    position_value+=price*order['amount']

        print(f"Testing Down\nAccount ID: {self.account.id}\nFinal asset: {self.account.asset}, position value: {position_value} Total:{self.account.asset+position_value}")

    
    def calculate_position(self):
        #頭寸大小 = 賬戶大小*賬戶風險(1%)/失效點
        position = (self.account.asset * 0.01) /self.risk_management
        return position
    
    #create(buy) a single order
    def buy(self,symbol):
        current_price = self.orginal_data[symbol]['Close'].iloc[self.current_index]
        # 計算總花費，包括手續費
        spend = self.calculate_position()
        #最小下單金額(based on binance API 5 USDT)
        if spend<5: 
            print("Buy order failed: Your account balance is insufficient")
            return
        # 計算手續費
        commission_paid = spend * self.account.commission
        # 計算實際用於購買資產的金額
        net_spend = spend - commission_paid
        # 計算購買的資產數量
        amount = net_spend / current_price
        print(f"symbol:{symbol},{self.orginal_data[symbol]['Datetime'][self.current_index]} price:{current_price} buy amount:{amount:.6f},spend:{net_spend}")
        '''
        更新帳戶資訊
        '''
        self.account.asset -= spend
        order_info = {
            'symbol':symbol,
            'Datetime': self.orginal_data[symbol]['Datetime'].iloc[self.current_index],
            'id':self.current_index,
            'buy_price': current_price,
            'amount': amount,
            'type': 'buy',
            'commission': commission_paid,
            'cost':spend,
            'status':False, #False: 尚未賣出 , True: 賣出
            'sell_time':"",
            "sell_price":0,
            'net value':0

        }
        self.account.orders_book.append(order_info)
        self.account.position =self.account.have_position()

    def compute_ROI(self,netProfit,netcost):
        return netProfit/netcost

    #sell single order
    def sell(self,order):

        if not order['status']:
            current_price = self.orginal_data[order['symbol']]['Close'].iloc[self.current_index]
            amount = order['amount']
            # 計算總收益
            gross_value = current_price * amount
            # 計算手續費
            commission_paid = gross_value * self.account.commission
            # 計算實際獲得的金額
            net_value = gross_value - commission_paid
            '''
            更新帳戶資訊
            '''
            self.account.asset += net_value
            order['sell_time'] = self.orginal_data[order['symbol']]['Datetime'].iloc[self.current_index]
            order['sell_price']=current_price
            order['net_value']=net_value
            order['status']=True
            
            #compute ROI
            roi = self.compute_ROI( (net_value-order['cost']),order['cost'])
            print(f"symbol:{order['symbol']},{order['sell_time']} price: {current_price} sell amount: {amount}, net gain: {net_value}, ROI:{roi:.3f}")
            self.account.position =self.account.have_position()
            
    
            

if __name__=='__main__':
    data_list =["C:\\Users\\louislin\\OneDrive\\桌面\\data_analysis\\backtesting_system\\data\\1h_BTC.csv","C:\\Users\\louislin\\OneDrive\桌面\\data_analysis\\backtesting_system\\data\\1h_ETH.csv","C:\\Users\\louislin\\OneDrive\桌面\\data_analysis\\backtesting_system\data\\1h_BNB.csv"]
    symbols = ["BTCUSD","ETHUSD","BNBUSD"]
    trader = backtest(0,data_list,symbols)
    trader.account.set_asset(1000)
    trader.account.set_commission(0.001)
    trader.run()