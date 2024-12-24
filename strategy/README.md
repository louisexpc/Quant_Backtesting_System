# README.md

# `strategy` 資料夾說明

### 資料夾功能

`strategy` 資料夾用於存放專案的交易策略邏輯及其相關配置。該專案目前以隨機指數相對強弱指標 (Stochastic RSI) 策略為核心，並支持多個交易對和參數的靈活設置。

---

## 檔案：`stoch_rsi.py`

### 檔案功能

`stoch_rsi.py` 實現了基於 Stochastic RSI 的交易策略邏輯，提供指標計算與交易信號生成功能。

---

### 類別：`StochRSI`

### 功能描述

`StochRSI` 是專案的策略核心類別，負責：

1. 根據配置檔載入策略參數。
2. 計算多種技術指標 (Stochastic RSI、EMA、MACD)。
3. 為每個交易對生成交易信號。

### 實作邏輯

1. **配置初始化**:
    - 從 `stoch_rsi.json` 中讀取配置參數，並將其應用於策略的指標計算和信號生成。
2. **數據處理**:
    - 透過 `data_transform` 方法將原始數據轉換為支持計算的格式。
    - 支援多個交易對 (如 BTCUSD、ETHUSD)。
3. **技術指標計算**:
    - 使用內部的指標類別（如 `StochasticRSI`, `ExponentialMovingAverage`, `MACD`）進行計算。
    - 每個交易對計算獨立的指標數據。
4. **信號生成邏輯**:
    - 買入信號 (Buy): 當 MACD 上穿訊號線且 Stochastic RSI 小於 20。
    - 賣出信號 (Sell): 當 MACD 下穿訊號線且 Stochastic RSI 大於 80。
    - 未有信號 (Hold): 不符合上述條件時。

---

### 方法詳解

### `__init__(self, original_data: dict)`

- **功能**: 初始化策略參數與數據處理。
- **參數**:
    - `original_data`: 包含多個交易對的原始數據字典。
- **邏輯**:
    - 載入配置檔。
    - 進行數據轉換，生成技術指標。

### `data_transform(self, original_data: dict) -> pd.DataFrame`

- **功能**: 將原始數據轉換為支持計算的格式。
- **輸出**: 每個交易對的收盤價數據，限制在配置的最大行數 (`limit`) 內。

### `compute_indicators(self) -> dict`

- **功能**: 計算 Stochastic RSI、EMA、MACD 等技術指標。
- **輸出**: 每個交易對的技術指標數據。

### `run(self) -> dict`

- **功能**: 生成每個交易對的交易信號。
- **輸出**:
    - `1`: 買入信號。
    - `1`: 賣出信號。
    - `0`: 無操作信號。

---

### 範例程式碼

### 初始化並生成交易信號

```python
python
複製程式碼
from strategy.stoch_rsi import StochRSI
import pandas as pd

# 模擬多交易對的原始數據
original_data = {
    "BTCUSD": pd.DataFrame({
        "Close": [30000, 30100, 29900, 29800, 29700]
    }),
    "ETHUSD": pd.DataFrame({
        "Close": [2000, 2010, 2020, 1990, 1980]
    })
}

# 初始化 Stochastic RSI 策略
stoch_rsi_strategy = StochRSI(original_data)

# 生成交易信號
signals = stoch_rsi_strategy.run()

print(signals)

```

### 執行結果

```python
python
複製程式碼
{
    "BTCUSD": 1,  # 買入信號
    "ETHUSD": 0   # 無操作信號
}

```

---

## 檔案：`stoch_rsi.json`

### 檔案功能

`stoch_rsi.json` 提供策略的參數配置，允許靈活調整策略行為。

---

### 配置結構

```json
json
複製程式碼
{
    "stoch_rsi": {
        "symbol": ["BTCUSD", "ETHUSD", "BNBUSD"],  // 支援的交易對
        "timeframe": "15m",                      // 時間框架
        "limit": 100,                            // 最大數據行數
        "param": {                               // 策略參數
            "stoch_period": 14,                 // 隨機指標周期
            "ema_period": 3,                    // EMA 移動平均線周期
            "macd_short_period": 5,             // MACD 短期線周期
            "macd_long_period": 35,             // MACD 長期線周期
            "macd_signal_period": 5             // MACD 訊號線周期
        }
    }
}

```

---

### 說明

1. **`symbol`**:
    - 指定交易對，如 `BTCUSD`、`ETHUSD`。
    - 可擴展至更多交易對。
2. **`timeframe`**:
    - 時間框架，如 `15m` 表示 15 分鐘的數據。
3. **`limit`**:
    - 最大數據行數，用於限制計算的數據範圍。
4. **`param`**:
    - 策略所需的技術指標參數：
        - `stoch_period`: Stochastic RSI 的計算周期。
        - `ema_period`: EMA 指標的計算周期。
        - `macd_short_period`: MACD 短期 EMA 的計算周期。
        - `macd_long_period`: MACD 長期 EMA 的計算周期。
        - `macd_signal_period`: MACD 訊號線的計算周期。

---

### 注意事項

1. **多交易對支持**:
    - 策略可同時處理多個交易對的數據，但每個交易對需有足夠的數據量。
2. **參數調整**:
    - 策略表現依賴參數的合理設置，需根據歷史數據進行優化。
3. **數據格式**:
    - 原始數據需包含 `Close` 欄位，否則會引發錯誤。
