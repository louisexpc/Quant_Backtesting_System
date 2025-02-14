import pandas as pd
import numpy as np
import os
import time

from pkg.ConfigLoader import config
from utils.indicator import ExponentialMovingAverage,MACD,StochasticRSI,RSI
from utils.trend_classification import trend_quantified


STRATEGY_CONFIG = "C:\\Users\\louislin\\OneDrive\\桌面\\data_analysis\\backtesting_system\\strategy\\macd_cross.json"
STRATEGY_NAME = "macd_cross"

class MACD_CROSS(object):
    def __init__(self, data_paths: list, symbols: list):
        self.symbols = symbols
        self.data_paths = data_paths
        self._idx = 0  # 統一的時間索引
        self.original_datas = self.loading_data()
        

        self.config = config(STRATEGY_CONFIG).load_config()[STRATEGY_NAME]
        self.param = self.config['param']
        self.limit = self.config['limit']

    def loading_data(self) -> dict:
        """Loads CSV files based on symbols and paths."""
        original_data = {}
        for i in range(len(self.data_paths)):
            try:
                original_data[self.symbols[i]] = pd.read_csv(self.data_paths[i])
            except FileNotFoundError as e:
                print(f"[Error] Unable to load file {self.symbols[i]} at {self.data_paths[i]}. Details: {e}")
        return original_data
    
    def generate_signal(self) -> dict:
        signals = {}
        for symbol in self.symbols:
            if symbol not in self.original_datas:
                raise ValueError(f"Market data for {symbol} not available.")
            
            data = self.original_datas[symbol]
            feature = pd.DataFrame()

            indicator = MACD(data, "Close", 
                            self.param['macd_short_period'], 
                            self.param['macd_long_period'], 
                            self.param['macd_signal_period'])

            feature["diff"] = indicator.get_macd_line()
            feature["dea"] = indicator.get_signal_line()

            def classify_signal(row: pd.Series):
                if row["diff"] is not None and row["dea"] is not None:
                    return float(row["diff"]) > float(row["dea"])
                return False  

            signals[symbol] = feature.apply(classify_signal, axis=1)

        return signals


            

            


class macd_cross(object):
    def __init__(self,original_data:dict):
        self.original_data = original_data
        
        '''
        loading config
        '''
        self.config = config(STRATEGY_CONFIG).load_config()[STRATEGY_NAME]
        self.symbols = self.config['symbol']
        self.param = self.config['param']
        self.limit = self.config['limit']
        self.data = self.data_transform(original_data)
        
        pass

    def data_transform(self,original_data:dict)->pd.DataFrame:
        data = pd.DataFrame()
        for symbol in self.symbols:
            df = original_data[symbol]
            if(df.size<=self.config['limit']):
                df_close = df['Close']
            else:
                df_close =df['Close'].iloc[:self.limit]
            data[symbol]=df_close.astype(float)
        #print(data)
        return data
    

    def run(self)->dict:
        """ Generate signal for every symbol """
        signal ={}
        for symbol in self.symbols:
            
            '''
            compute macd
            '''
            data = self.data[symbol].to_frame(name =symbol)
           
            indicator=MACD(data,symbol,self.param['macd_short_period'],self.param['macd_long_period'],self.param['macd_signal_period'])
            diff = indicator.get_macd_line()
            dea = indicator.get_signal_line()
            long_ema = indicator.get_long_ema()
            mid_ema = ExponentialMovingAverage(data,symbol,self.param['ema_period']).get_ema()
            short_ema = indicator.get_short_ema()
            moment = diff-dea
            # cross
            #ema_cross_down = diff.iloc[-1]<0 and ((short_ema.iloc[-1] - long_ema.iloc[-1])<=0 and (short_ema.iloc[-2] - long_ema.iloc[-2])>=0)
            #ema_cross_up = diff.iloc[-1]>0 and ((short_ema.iloc[-1] - long_ema.iloc[-1])>=0 and (short_ema.iloc[-2] - long_ema.iloc[-2])<=0)
            # ema_cross_up = np.sign(short_ema.iloc[-1] - long_ema.iloc[-1]) > np.sign(short_ema.iloc[-2] - long_ema.iloc[-2])
            # ema_cross_down = np.sign(short_ema.iloc[-1] - long_ema.iloc[-1]) < np.sign(short_ema.iloc[-2] - long_ema.iloc[-2])
            ema_cross_up = (np.sign(short_ema.iloc[-1] - long_ema.iloc[-1]) > np.sign(short_ema.iloc[-2] - long_ema.iloc[-2]) 
                and diff.iloc[-1] > 0)
            ema_cross_down = (np.sign(short_ema.iloc[-1] - long_ema.iloc[-1]) < np.sign(short_ema.iloc[-2] - long_ema.iloc[-2]) 
                            and diff.iloc[-1] < 0)

            #moment_cross_down = moment.iloc[-1]<0 and moment.iloc[-2]>0 #紅牙轉黑牙
            #moment_corss_up = moment.iloc[-1]>0 and moment.iloc[-2]<0 #黑牙轉紅牙
            # moment_cross_up = np.sign(moment.iloc[-1]) > np.sign(moment.iloc[-2])
            # moment_cross_down = np.sign(moment.iloc[-1]) < np.sign(moment.iloc[-2])
            threshold = 0.5
            moment_cross_up = (np.sign(moment.iloc[-1]) > np.sign(moment.iloc[-2]) 
                   and abs(moment.iloc[-1] - moment.iloc[-2]) > threshold)
            moment_cross_down = (np.sign(moment.iloc[-1]) < np.sign(moment.iloc[-2]) 
                                and abs(moment.iloc[-1] - moment.iloc[-2]) > threshold)



            # trend classification
            short_slope = short_ema.diff().iloc[-1]
            mid_slope = mid_ema.diff().iloc[-1]
            long_slope = long_ema.diff().iloc[-1]
            uptrend = short_ema.iloc[-1] > mid_ema.iloc[-1] > long_ema.iloc[-1] and short_slope > 0 and mid_slope > 0
            downtrend = short_ema.iloc[-1] < mid_ema.iloc[-1] < long_ema.iloc[-1] and short_slope < 0 and mid_slope < 0


            if uptrend:
                # 多頭
                if ema_cross_up and moment.iloc[-1]>0:
                    signal[symbol] = 1
                elif ema_cross_down and moment_cross_down and diff.iloc[-1] < 0:
                    signal[symbol] = -1
                elif moment.iloc[-1]>=0 and moment.iloc[-1]<=moment.iloc[-2] and moment.iloc[-2] <= moment.iloc[-3]:
                    signal[symbol] = -1
                elif moment.iloc[-1]<=0:
                    signal[symbol] = 0
                else:
                    signal[symbol]=0
                pass
            elif downtrend:
                #空頭
                if ema_cross_down and moment.iloc[-1]<0:
                    signal[symbol]=-1
                elif moment_cross_down:
                    signal[symbol]=-1
                elif ema_cross_up and moment_cross_up and diff.iloc[-1]>0:
                    signal[symbol] = 1
                elif ema_cross_up :
                    if moment.iloc[-1]>= moment.iloc[-2] and moment.iloc[-2] >= moment.iloc[-3]:
                        signal[symbol]=1
                else:
                    signal[symbol]=0
                pass
            else:
                #震盪區間
                signal[symbol]=0
                pass
        return signal
            
