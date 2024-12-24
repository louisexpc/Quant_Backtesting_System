import pandas as pd

def sharp_ratio(net_profit: pd.Series, total_cost: pd.Series, Rf: float) -> float:
    '''
    Equation: sharp ratio = \\frac{E(R_p) - R_f}{\\sigma_p}
    '''
    Rp = (net_profit ) / total_cost
    Expectation = Rp.mean()
    variance = ((Rp - Expectation) ** 2).sum() / (Rp.size - 1)
    std = variance ** 0.5 
    sharpe = (Expectation - Rf) / std
    return float(sharpe)

def profit_loss_ratio(net_profit: pd.Series) -> float:
    
    average_gain = net_profit[net_profit > 0].mean()
    
    average_loss = abs(net_profit[net_profit < 0].mean()) if net_profit[net_profit < 0].size > 0 else 0
   
    PL_ratio = average_gain / average_loss if average_loss != 0 else float('inf')  # 如果沒有損失則返回無窮大
    return float(PL_ratio)


def win_rate(net_profit: pd.Series) -> float:
    total_trade_number = net_profit.size
    if total_trade_number == 0:
        return 0.0
    profit_trade_number = net_profit[net_profit > 0].size
    return float(profit_trade_number / total_trade_number)

def maximum_drawdown(buy_idx:int , sell_idx:int , close:pd.DataFrame)->float:
    '''
    Cumulative Max(t)=max(Price(t),Price(t-1),…,Price(1))
    Drawdown(t)= (Price(t)-Cumulative Max(t))/ Cumulative Max(t)
    Maximum Drawdown (MDD)=min(Drawdown(t))
    '''

    # 取得指定日期範圍內的收盤價
    price_data = close.loc[buy_idx:sell_idx]
    
    # 計算累積最大價格
    cumulative_max = price_data.cummax()
    
    # 計算每個日期的回撤
    drawdowns = (price_data - cumulative_max) / cumulative_max
    
    # 計算最大回撤
    max_drawdown = drawdowns.min()
    
    return max_drawdown

def roi(net_profit:float,totol_cost:float)->float:
    return net_profit/totol_cost