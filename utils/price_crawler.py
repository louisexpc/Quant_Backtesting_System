import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
def fetch_all_ohlcv( symbols:list, timeframe:str, since:int, limit=1000)->None:
    """
    獲取 Binance 的完整 OHLCV 資料
    :param exchange: ccxt 交易所對象
    :param symbol: 交易對，例如 'BTC/USDT'
    :param timeframe: 時間週期，例如 '1m', '1h', '1d'
    :param since: 起始時間（Unix 毫秒）
    :param limit: 每次請求的最大條數，默認為 1000
    :return: 包含所有數據的 DataFrame
    """
    exchange = ccxt.binance()
    for symbol in symbols:
        all_data = []  # 用於存儲所有資料
        current_since = since  # 當前請求的起始時間

        while True:
            # 獲取 OHLCV 資料
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)
            
            if not ohlcv:
                break  # 如果沒有更多數據，結束迴圈
            
            all_data.extend(ohlcv)  # 添加數據到總集合中
            
            # 更新下一次請求的起始時間
            current_since = ohlcv[-1][0] + 1  # 避免重複數據

            print(f"Fetched {len(ohlcv)} records from {datetime.utcfromtimestamp(current_since / 1000)}")

            # 如果返回的數據少於 limit，說明已到達最後
            if len(ohlcv) < limit:
                break

        # 格式化並轉換時間戳為 UTC+8
        df = pd.DataFrame(all_data, columns=['Datetime','Open','High','Low','Close','Volume'])
        df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms')  # 格式化時間戳
        df['Datetime'] = df['Datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)

        #儲存資料
        if df.isnull().any().any():
            print(f"Query {symbol} missing!")
        else:
            df.to_csv(rf".\data\{timeframe}_{symbol}.csv",index=False)
        time.sleep(0.5)

# 初始化 Binance

# 配置參數
symbols = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT"]  # 交易對
timeframe = '1d'     # 時間週期
since = int(datetime(2022, 1, 1).timestamp() * 1000)  # 起始時間（yyyy-mm-dd）

# 獲取完整資料
data = fetch_all_ohlcv(symbols, timeframe, since)

