import yfinance as yf
import pandas as pd
'''
period: data period to download (either use period parameter or use start and end) Valid periods are:
“1d”, “5d”, “1mo”, “3mo”, “6mo”, “1y”, “2y”, “5y”, “10y”, “ytd”, “max”
interval: data interval (1m data is only for available for last 7 days, and data interval <1d for the last 60 days) Valid intervals are:
“1m”, “2m”, “5m”, “15m”, “30m”, “60m”, “90m”, “1h”, “1d”, “5d”, “1wk”, “1mo”, “3mo”
'''

""" latest price"""
# data = yf.download(
#     tickers = "BTC-USD",
#     period = "1mo",
#     interval = "1h"
# )
""" specific date price """

""" data = yf.download(
    tickers = "ETH-USD",
    start='2024-10-24',
    end='2024-11-24',
    interval = "1h"
) """

def price_crawler(symbols:list,interval:str,start:str,end:str):
    for symbol in symbols:
        data = yf.download(
            tickers=symbol,
            start=start,
            end=end,
            interval=interval
        )
        if data.isnull().any().any():
            print("data missing")
            return
        else:
            data.to_csv(f"./data/{interval}_{symbol}.csv")

symbols = ["BTC-USD","ETH-USD","BNB-USD"]
start = "2024-10-24"
end = '2024-11-24'
interval = "1h"
price_crawler(symbols,interval,start,end)
