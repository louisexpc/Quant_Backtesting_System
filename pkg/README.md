# README.md

# `pkg` 資料夾說明

### 資料夾功能

`pkg` 資料夾主要負責專案中的配置管理，提供統一的配置檔讀取工具，並且以單例模式 (Singleton) 確保每個配置檔的唯一性，避免重複載入造成不必要的資源浪費。

---

## 檔案：`ConfigLoader.py`

### 檔案功能

`ConfigLoader.py` 是一個配置管理工具，主要用於載入 JSON 格式的配置檔，並以物件的形式提供讀取功能，確保配置資料的一致性與可重用性。

---

### 類別：`config`

### 功能描述

`config` 類別實現了一個通用的配置載入工具，支援單例模式，確保每個配置檔在專案中只會被載入一次。

### 實作邏輯

1. **單例模式**:
    - 使用 `__new__` 方法來實現單例模式，確保對同一路徑的配置檔只會有一個實例。
    - 所有配置檔的實例都存放在類別變數 `_instance` 中。
2. **配置檔讀取**:
    - 提供 `load_config` 方法，用於讀取 JSON 格式的配置檔並返回對應的字典物件。

### 重要屬性

- **`_instance`**: 類別變數，存放所有已載入的配置檔實例。
- **`config_file_path`**: 實例變數，記錄配置檔路徑。
- **`config`**: 實例變數，存放已載入的配置資料。

### 方法詳解

### `__new__(cls, config_path)`

- **功能**: 確保每個配置檔只會有一個實例。
- **參數**:
    - `config_path`: 配置檔的路徑。
- **回傳值**: 該路徑對應的 `config` 類別實例。
- **邏輯**:
    - 如果該路徑已存在於 `_instance`，直接返回對應的實例。
    - 否則，創建新實例並存入 `_instance`。

### `load_config(self)`

- **功能**: 載入配置檔並返回其內容。
- **邏輯**:
    - 如果 `config` 為 `None`，則從 `config_file_path` 載入 JSON 檔。
    - 返回載入的配置資料。

---

### 範例程式碼

### 初始化並載入配置

```python
python
複製程式碼
from pkg.ConfigLoader import config

# 初始化配置物件，指定配置檔路徑
strategy_config = config("./strategy/stoch_rsi.json")

# 載入配置資料
config_data = strategy_config.load_config()

print(config_data)

```

### 執行結果

假設 `stoch_rsi.json` 的內容如下：

```json
json
複製程式碼
{
    "stoch_rsi": {
        "symbol": ["BTCUSD", "ETHUSD", "BNBUSD"],
        "timeframe": "15m",
        "limit": 100,
        "param": {
            "stoch_period": 14,
            "ema_period": 3,
            "macd_short_period": 5,
            "macd_long_period": 35,
            "macd_signal_period": 5
        }
    }
}

```

程式輸出：

```python
python
複製程式碼
{
    "stoch_rsi": {
        "symbol": ["BTCUSD", "ETHUSD", "BNBUSD"],
        "timeframe": "15m",
        "limit": 100,
        "param": {
            "stoch_period": 14,
            "ema_period": 3,
            "macd_short_period": 5,
            "macd_long_period": 35,
            "macd_signal_period": 5
        }
    }
}

```

---

### 注意事項

1. **單例模式**:
    - 當多次初始化相同路徑的配置物件時，會返回相同的實例。
    - 修改一個實例的屬性會影響所有指向該實例的變數。
2. **JSON 格式要求**:
    - 配置檔需為合法的 JSON 格式。
    - 路徑需正確，否則會引發檔案讀取錯誤。
3. **多執行緒安全性**:
    - 此實作未提供多執行緒安全性保證，如需應用於多執行緒環境，需額外處理同步問題。