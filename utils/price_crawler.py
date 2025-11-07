import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_all_ohlcv( symbols:list, timeframe:str, since:int, limit=1000)->list:
    """
    獲取 Binance 的完整 OHLCV 資料
    :param exchange: ccxt 交易所對象
    :param symbol: 交易對，例如 'BTC/USDT'
    :param timeframe: 時間週期，例如 '1m', '1h', '1d'
    :param since: 起始時間（Unix 毫秒）
    :param limit: 每次請求的最大條數，默認為 1000
    :return: 包含所有數據的 DataFrame

    return: list, paths of the  crawled data
    """
    data_paths=[]
    exchange = ccxt.binance({
        'options': {
            'defaultType': 'future',  # 使用永續合約市場
        }
    })
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
            path = rf".\data\{timeframe}_{symbol}.csv"
            data_paths.append(path)
            df.to_csv(path,index=False)
        time.sleep(0.5)
    return data_paths

def fetch_all_funding_rates(symbols: list, since: int, limit=1000) -> None:
    """
    爬取 Binance 永續合約的歷史資金費率數據，並存為 CSV。
    :param symbols: 交易對列表，例如 ['BTCUSDT', 'ETHUSDT']
    :param since: 起始時間（Unix 毫秒）
    :param limit: 每次請求的最大條數，默認為 1000
    """
    exchange = ccxt.binance()
    
    for symbol in symbols:
        base_symbol = symbol.replace("USDT", "")  # 確保正確格式化
        formatted_symbol = f"{base_symbol}/USDT:USDT"  # 轉換為 ccxt 交易對格式
        all_data = []  # 存儲所有數據
        current_since = since  # 當前請求的起始時間
        
        while True:
            try:
                # 獲取資金費率歷史數據
                funding_rates = exchange.fetch_funding_rate_history(formatted_symbol, since=current_since, limit=limit)
                
                if not funding_rates:
                    break  # 沒有更多數據則結束迴圈
                
                all_data.extend(funding_rates)
                
                # 更新下一次請求的起始時間，避免重複數據
                current_since = funding_rates[-1]['timestamp'] + 1
                
                print(f"Fetched {len(funding_rates)} records from {datetime.utcfromtimestamp(current_since / 1000)}")
                
                # 如果返回的數據少於 limit，說明已到達最後
                if len(funding_rates) < limit:
                    break
                
                time.sleep(0.5)  # 避免 API 限制
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                break
        
        # 轉換數據格式
        if all_data:
            df = pd.DataFrame(all_data)
            df['Datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['Datetime'] = df['Datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
            df['FundingRate'] = df['fundingRate'].astype(float)
            
            # 過濾需要的欄位
            df = df[['Datetime', 'FundingRate']]
            
            # 儲存 CSV 文件
            filename = rf"./data/funding_rate_{symbol}.csv"
            df.to_csv(filename, index=False)
            print(f"Saved {symbol} funding rate data to {filename}")
        else:
            print(f"No data for {symbol}")
# 測試函數執行
if __name__ == "__main__":
    test_symbols = ["BTCUSDT", "ETHUSDT"]
    test_since = int(datetime(2024, 1, 1).timestamp() * 1000)
    fetch_all_funding_rates(test_symbols, test_since)


    # # 配置參數
    # symbols = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT"]  # 交易對
    # symbols = ["BTCUSDT"]  # 交易對

    # timeframe = '1d'     # 時間週期
    # since = int(datetime(2025, 1, 1).timestamp() * 1000)  # 起始時間（yyyy-mm-dd）

    # # 獲取完整資料
    # data = fetch_all_ohlcv(symbols, timeframe, since)

