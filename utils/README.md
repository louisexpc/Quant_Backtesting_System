# `utils` 資料夾說明

### 資料夾功能

`utils` 資料夾包含專案中核心的輔助功能模組，包括帳戶管理、訂單操作、技術指標計算以及績效評估工具，為交易策略提供基礎功能支持。

---

## 檔案：`account.py`

### 檔案功能

管理交易帳戶，包括資金操作、手續費設定與交易歷史記錄。

---

### 類別：`Account`

### 功能描述

`Account` 類別負責:

1. 記錄帳戶資產與手續費。
2. 提供交易歷史的接口，支援訂單記錄與查詢。

### 重要屬性

- **`id`**: 帳戶的唯一識別碼。
- **`asset`**: 帳戶內可用資金。
- **`commission`**: 每筆交易的手續費比率。
- **`history`**: 記錄所有交易的訂單歷史 (使用 `order_history` 類別)。

### 方法詳解

### `set_asset(self, cash)`

- **功能**: 設定帳戶的初始資金。
- **參數**: `cash` 為帳戶資產數量。

### `set_commission(self, commission)`

- **功能**: 設定交易手續費比例。
- **參數**: `commission` 為手續費比率 (如 0.001 表示千分之一)。

### 使用範例

```python
python
複製程式碼
from utils.account import Account

account = Account(1)
account.set_asset(1000)
account.set_commission(0.001)

print(account.asset)  # 1000
print(account.commission)  # 0.001

```

---

## 檔案：`order.py`

### 檔案功能

提供交易訂單管理工具，包括單筆訂單與歷史記錄的管理。

---

### 類別：`order`

### 功能描述

`order` 類別用於表示單筆交易訂單，包含訂單的詳細資訊。

### 初始化參數

- **`info`**: 一個字典，包含訂單的基本資訊，如交易對、買入價格、數量等。

---

### 類別：`order_history`

### 功能描述

`order_history` 負責管理所有的交易訂單歷史，支援以下功能：

1. 新增訂單。
2. 查詢特定訂單。
3. 更新訂單資訊。
4. 評估交易績效。

### 方法詳解

### `update_history(self, new_order: order)`

- **功能**: 新增訂單到交易記錄。

### `search_positions(self) -> pd.DataFrame`

- **功能**: 查詢尚未平倉的訂單。

### `evaluation(self)`

- **功能**: 評估交易策略績效，計算 Sharpe 比率、盈虧比、勝率等。

### 使用範例

```python
python
複製程式碼
from utils.order import order, order_history

info = {
    "symbol": "BTCUSD",
    "buy_price": 30000,
    "amount": 0.1,
    "total_cost": 3000,
    "status": False
}

new_order = order(info)
history = order_history()
history.update_history(new_order)
history.evaluation()

```

---

## 檔案：`indicator.py`

### 檔案功能

提供技術指標的計算工具，包括:

1. 簡單移動平均線 (SMA)
2. 指數移動平均線 (EMA)
3. 隨機相對強弱指標 (Stochastic RSI)
4. 布林通道 (Bollinger Bands)
5. 平滑異同移動平均線 (MACD)

---

### 類別與功能

### `SmoothMovingAverage`

- **功能**: 計算簡單移動平均線 (SMA)。
- **參數**:
    - `data`: 包含價格資訊的 DataFrame。
    - `symbol`: 計算的目標欄位名稱。
    - `window`: 移動平均的窗口大小。

### `ExponentialMovingAverage`

- **功能**: 計算指數移動平均線 (EMA)。

### `StochasticRSI`

- **功能**: 計算隨機 RSI 指標。

### `MACD`

- **功能**: 計算 MACD 指標及其訊號線與柱狀圖。

### 使用範例

```python
python
複製程式碼
from utils.indicator import StochasticRSI, ExponentialMovingAverage, MACD
import pandas as pd

data = pd.DataFrame({"Close": [30000, 30100, 29900, 29800, 29700]})

# 計算 Stochastic RSI
stoch_rsi = StochasticRSI(data, "Close")
print(stoch_rsi.get_stochastic_rsi())

# 計算 EMA
ema = ExponentialMovingAverage(data, "Close", window=3)
print(ema.get_ema())

```

---

## 檔案：`evalution.py`

### 檔案功能

提供策略績效的評估方法，支援以下指標:

1. **Sharpe 比率**: 衡量單位風險下的報酬。
2. **盈虧比**: 平均獲利與損失的比率。
3. **勝率**: 獲利交易數量佔總交易數量的比例。
4. **最大回撤 (MDD)**: 衡量最大資金回撤幅度。
5. **投資報酬率 (ROI)**: 計算獲利與投入資金的比率。

---

### 方法詳解

### `sharp_ratio(net_profit, total_cost, Rf)`

- **功能**: 計算 Sharpe 比率。
- **參數**:
    - `net_profit`: 淨收益序列。
    - `total_cost`: 投資成本序列。
    - `Rf`: 無風險利率。

### `profit_loss_ratio(net_profit)`

- **功能**: 計算盈虧比。

### `win_rate(net_profit)`

- **功能**: 計算勝率。

### `maximum_drawdown(buy_idx, sell_idx, close)`

- **功能**: 計算最大回撤。

### 使用範例

```python
python
複製程式碼
from utils.evalution import sharp_ratio, profit_loss_ratio, win_rate

net_profit = pd.Series([100, -50, 200, -100])
total_cost = pd.Series([1000, 500, 2000, 1000])
Rf = 0.008

# 計算 Sharpe 比率
print(sharp_ratio(net_profit, total_cost, Rf))

# 計算盈虧比
print(profit_loss_ratio(net_profit))

# 計算勝率
print(win_rate(net_profit))

```

---

### 注意事項

1. **資料格式**:
    - 計算指標時，數據需為合法的 pandas DataFrame 或 Series 格式。
2. **多交易對支持**:
    - 儘管模組主要針對單交易對設計，但可以擴展至多交易對分析。
3. **技術指標參數調整**:
    - 建議根據策略需求動態調整技術指標的參數，以獲得最佳效果。
