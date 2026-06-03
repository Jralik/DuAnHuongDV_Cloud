<img width="492" height="118" alt="image" src="https://github.com/user-attachments/assets/d1e1713c-bc69-4c12-b74e-9506923a277e" /># 🤖 AI Trading System — Model Documentation

Tài liệu mô tả toàn bộ pipeline học máy của hệ thống, từ thu thập dữ liệu thô (raw), qua feature engineering, đến training và dự đoán bằng mô hình Ensemble.

---

## 📐 Tổng quan Kiến trúc

```
         Sàn giao dịch (Binance / OKX / Kraken)
                         │
                         ▼
              [01] DATA INGESTION (Bronze)
              Thu thập OHLCV → Delta Lake Bronze
                         │
                         ▼
          [02] FEATURE ENGINEERING (Silver)
          Tính Technical Indicators + ML Features
          → Delta Lake Silver
                         │
                         ▼
              [03] MODEL TRAINING (Gold)
          XGBoost  +  Random Forest  +  LSTM
          → MLflow Tracking (Databricks)
                         │
                         ▼
            [04] ENSEMBLE PREDICTION
          Weighted Voting → Tín hiệu MUA / BÁN / GIỮ
```

---

## 1. 📥 Data (Dữ liệu)

### 1.1 Nguồn dữ liệu

| Thuộc tính     | Giá trị                                              |
|----------------|------------------------------------------------------|
| **Symbols**    | `BTC/USDT`, `ETH/USDT`                               |
| **Timeframe**  | `1h` (nến 1 giờ)                                     |
| **Lịch sử**   | Mặc định 90 ngày (~2.160 nến/symbol)                 |
| **Nguồn**      | CCXT (ưu tiên OKX → Binance → Kraken → KuCoin)      |
| **Schema**     | `timestamp`, `open`, `high`, `low`, `close`, `volume`, `symbol`, `timeframe` |

### 1.2 Kiến trúc Delta Lake (Medallion Architecture)

```
Bronze Layer (ohlcv_bronze)        Silver Layer (ohlcv_silver)
───────────────────────────        ─────────────────────────────────────────
  Dữ liệu OHLCV thô                 OHLCV + 50+ Technical Indicators
  Partitioned by: symbol/timeframe  + 60+ ML Features + target labels
  Upsert bằng MERGE (no duplicate)  Partitioned by: symbol/timeframe
```

### 1.3 Chất lượng dữ liệu (Data Quality)

Trước khi ghi vào Delta Lake, hệ thống kiểm tra:
- Tỷ lệ missing values ≤ 5%
- Tỷ lệ `volume = 0` ≤ 10%
- Số nến tối thiểu để train model: **500 nến**

### 1.4 Target Label (Nhãn dự đoán)

Hệ thống dự đoán **hướng giá sau 24 giờ tới** (target_periods = 24) bằng cách sử dụng **Dynamic Threshold (Ngưỡng động dựa trên ATR)** thay vì mức phần trăm cố định. Điều này giúp mô hình tự thích nghi với trạng thái thị trường (sideway hay bão tố).

#### 🧮 Cách tính toán ATR (Average True Range) và Ngưỡng động:

1. **True Range ($TR_t$)** của nến hiện tại được tính bằng giá trị lớn nhất trong 3 khoảng cách sau:
   $$TR_t = \max \left( High_t - Low_t, \, \left| High_t - Close_{t-1} \right|, \, \left| Low_t - Close_{t-1} \right| \right)$$
   *Trong đó: $High_t$, $Low_t$ là giá cao nhất, thấp nhất của nến hiện tại; $Close_{t-1}$ là giá đóng cửa nến trước.*

2. **Average True Range ($ATR_t$)** chu kỳ 14 được làm mịn theo phương pháp của Wilder:
   $$ATR_t = \frac{ATR_{t-1} \times 13 + TR_t}{14}$$

3. **Tỷ lệ phần trăm ATR (`atr_pct`)** so với giá đóng cửa để chuẩn hóa biến động:
   $$\text{ATRPct}_t = \frac{ATR_t}{Close_t}$$

4. **Ngưỡng biến động động (`dynamic_threshold`)** được tính bằng trung bình trượt 24 chu kỳ của `atr_pct`:
   <img width="492" height="118" alt="image" src="https://github.com/user-attachments/assets/ba69c157-6329-45d3-9c0f-8f71a609dae4" />
   *(Nếu chưa đủ 24 nến để tính trung bình trượt, hệ thống sẽ sử dụng giá trị mặc định là $0.005$ hay $0.5\%$).*

- Ngưỡng biến động động (`dynamic_threshold`) = `ATR % trung bình 24 nến * 1.0`

| Label | Ý nghĩa | Điều kiện |
|-------|---------|-----------|
| `1`   | **Tăng mạnh (Breakout)** | Return 24h tới > `dynamic_threshold` |
| `0`   | **Không tăng mạnh** | Return 24h tới ≤ `dynamic_threshold` |

> **`is_valid_target`**: Cột boolean đánh dấu những mẫu có xu hướng rõ (tăng **hoặc** giảm mạnh vượt `dynamic_threshold`), dùng để lọc nhiễu sideway khi train XGBoost/Random Forest.

---

## 2. 🔧 Feature Engineering (Đặc trưng)

Pipeline feature engineering chạy **2 tầng song song**:
- **Local** (`core/features.py`): Phục vụ prediction real-time và train local
- **Databricks** (`databricks_pipeline/02_feature_engineering.py`): Phục vụ train trên cloud (đồng bộ hoàn toàn)

### 2.1 Technical Indicators (Chỉ báo kỹ thuật gốc)

Đây là các chỉ báo được tính từ OHLCV và trở thành đầu vào cho feature engineering:

#### 📈 Trend (Xu hướng)
| Feature | Tham số | Mô tả |
|---------|---------|-------|
| `sma_10`, `sma_20`, `sma_50` | periods: 10, 20, 50 | Simple Moving Average |
| `ema_9`, `ema_21`, `ema_55`, `ema_200` | periods: 9, 21, 55, 200 | Exponential Moving Average |
| `macd`, `macd_signal`, `macd_histogram` | fast=12, slow=26, signal=9 | MACD & Signal Line |
| `adx`, `di_plus`, `di_minus` | period=14 | Average Directional Index |

#### ⚡ Momentum (Động lượng)
| Feature | Tham số | Mô tả |
|---------|---------|-------|
| `rsi` | period=14 | Relative Strength Index (0–100) |
| `stoch_k`, `stoch_d` | k=14, d=3 | Stochastic Oscillator |
| `cci` | period=20 | Commodity Channel Index |
| `willr` | period=14 | Williams %R |

#### 🌊 Volatility (Biến động)
| Feature | Tham số | Mô tả |
|---------|---------|-------|
| `bb_upper`, `bb_mid`, `bb_lower` | period=20, std=2 | Bollinger Bands |
| `bb_bandwidth`, `bb_percent` | — | Độ rộng & %B của BB |
| `atr` | period=14 | Average True Range |

#### 📊 Volume (Khối lượng)
| Feature | Tham số | Mô tả |
|---------|---------|-------|
| `obv` | — | On-Balance Volume |
| `mfi` | period=14 | Money Flow Index |

#### 🎯 Custom
| Feature | Mô tả |
|---------|-------|
| `price_position` | Vị trí giá trong kênh giá 20 phiên (0=đáy, 1=đỉnh) |
| `momentum_score` | Điểm momentum tổng hợp [-100, +100] (RSI×0.3 + MACD×0.25 + Stoch×0.2 + CCI×0.15 + ADX×0.1) |

---

### 2.2 ML Features (Đặc trưng cho mô hình ML)

Được tính từ các indicators và dữ liệu OHLCV gốc.

#### 📍 Group 1: Price Features (9 features)
| Feature | Mô tả |
|---------|-------|
| `price_sma20_ratio` | `close / sma_20` — khoảng cách giá với MA trung hạn |
| `price_sma50_ratio` | `close / sma_50` — khoảng cách giá với MA dài hạn |
| `price_ema21_ratio` | `close / ema_21` — khoảng cách giá với EMA nhanh |
| `sma10_sma20_cross` | `(sma_10 - sma_20) / close` — tín hiệu golden/death cross |
| `high_low_range` | `(high - low) / close` — biên độ dao động trong nến |
| `close_open_range` | `(close - open) / close` — sức mạnh thân nến |
| `bb_pct_b` | Vị trí giá trong dải Bollinger (0=band dưới, 1=band trên) |
| `bb_width` | `(bb_upper - bb_lower) / close` — độ rộng dải BB |

#### 📉 Group 2: Return Features (9 features)
| Feature | Mô tả |
|---------|-------|
| `return_1`, `return_3`, `return_6`, `return_12`, `return_24` | Tỷ suất lợi nhuận đơn sau 1/3/6/12/24 nến |
| `log_return_1` | Logarit tỷ suất lợi nhuận 1 nến |
| `cum_return_6`, `cum_return_12`, `cum_return_24` | Tổng lợi nhuận tích lũy 6/12/24 nến |

#### 📊 Group 3: Rolling Statistics (8 features)
| Feature | Mô tả |
|---------|-------|
| `volatility_6`, `volatility_12`, `volatility_24` | Độ lệch chuẩn của return trong cửa sổ 6/12/24 nến |
| `zscore_6`, `zscore_12`, `zscore_24` | Z-score của giá so với trung bình cửa sổ |
| `skew_20` | Độ lệch (skewness) phân phối return trong 20 nến |
| `kurt_20` | Độ nhọn (kurtosis) phân phối return trong 20 nến — đo rủi ro đuôi béo |

#### 🕐 Group 4: Lagged Features (18 features)
Các chỉ báo quan trọng được trích xuất ở các độ trễ 1, 3, 6, 12 nến:

| Nguồn | Lag 1 | Lag 3 | Lag 6 | Lag 12 |
|-------|-------|-------|-------|--------|
| `rsi` | `rsi_lag_1` | `rsi_lag_3` | `rsi_lag_6` | `rsi_lag_12` |
| `macd_histogram` | `macd_histogram_lag_1` | `macd_histogram_lag_3` | `macd_histogram_lag_6` | `macd_histogram_lag_12` |
| `momentum_score` | `momentum_score_lag_1` | `momentum_score_lag_3` | `momentum_score_lag_6` | `momentum_score_lag_12` |
| `adx` | `adx_lag_1` | `adx_lag_3` | `adx_lag_6` | `adx_lag_12` |

Thêm:
| Feature | Mô tả |
|---------|-------|
| `rsi_change_3` | `rsi - rsi_lag_3` — tốc độ thay đổi RSI (phát hiện divergence) |
| `rsi_change_6` | `rsi - rsi_lag_6` |

#### 🕯️ Group 5: Candlestick Features (6 features)
| Feature | Mô tả |
|---------|-------|
| `body_size` | `abs(close - open) / close` — độ lớn thân nến |
| `upper_shadow` | Chiều dài râu nến trên (lực bán từ vùng cao) |
| `lower_shadow` | Chiều dài râu nến dưới (lực bắt đáy) |
| `is_bullish` | `1` nếu nến tăng (close > open), `0` nếu giảm |
| `bullish_streak` | Số nến xanh liên tiếp (đếm tích lũy, reset khi đổi màu) |
| `bearish_streak` | Số nến đỏ liên tiếp |

#### 📦 Group 6: Volume Features (6 features)
| Feature | Mô tả |
|---------|-------|
| `volume_zscore` | Z-score của khối lượng so với trung bình 20 nến |
| `volume_sma20_ratio` | Tỷ lệ khối lượng so với trung bình 20 nến |
| `volume_change_3` | Tốc độ tăng trưởng khối lượng 3 nến |
| `obv_change_12` | % thay đổi OBV trong 12 nến (phát hiện dòng tiền thông minh) |
| `dollar_volume` | `close × volume` — giá trị giao dịch theo USD |
| `dollar_volume_change` | % thay đổi dollar volume 6 nến |

#### ⚡ Group 7: Momentum & Oscillator Features (11 features)
| Feature | Mô tả |
|---------|-------|
| `rsi_oversold` | `1` nếu RSI < 30 (quá bán) |
| `rsi_overbought` | `1` nếu RSI > 70 (quá mua) |
| `rsi_neutral` | `1` nếu RSI trong [40, 60] |
| `rsi_normalized` | `(rsi - 50) / 50` — chuẩn hóa RSI về [-1, +1] |
| `stoch_diff` | `stoch_k - stoch_d` — khoảng cách hai đường Stochastic |
| `stoch_oversold` | `1` nếu Stoch_K < 20 |
| `stoch_overbought` | `1` nếu Stoch_K > 80 |
| `macd_cross` | `sign(macd - macd_signal)` — hướng giao cắt MACD (+1/-1) |
| `macd_cross_change` | Sự thay đổi trạng thái giao cắt — bắt điểm vào lệnh chính xác |
| `mfi_oversold` | `1` nếu MFI < 20 |
| `mfi_overbought` | `1` nếu MFI > 80 |
| `mfi_normalized` | `(mfi - 50) / 50` |

#### 📐 Group 8: Trend Confirmation Features (11 features)
| Feature | Mô tả |
|---------|-------|
| `adx_strong` | `1` nếu ADX > 25 (có trend rõ ràng) |
| `adx_slope` | `adx.diff(3)` — tốc độ tăng/giảm của sức mạnh trend |
| `ema_alignment` | `sign(ema_9 - ema_21)` — EMA nhanh nằm trên/dưới EMA chậm |
| `ema_alignment_change` | Sự thay đổi alignment (phát hiện đảo chiều trend) |
| `price_to_ema200` | `close / ema_200` — khoảng cách giá so với đường cản siêu mạnh |
| `trend_macro` | `1` nếu `close > ema_200` (uptrend dài hạn) |
| `higher_high_12` | `1` nếu giá tạo đỉnh cao hơn đỉnh 12 nến trước (Dow Theory) |
| `lower_low_12` | `1` nếu giá tạo đáy thấp hơn đáy 12 nến trước |
| `roc_6` | Rate of Change 6 nến (% thay đổi giá thuần túy) |
| `roc_12` | Rate of Change 12 nến |
| `roc_24` | Rate of Change 24 nến |

---

### 2.3 Tổng hợp Features

| Nhóm | Số features |
|------|------------|
| Technical Indicators (raw) | ~25 |
| Price Features | 8 |
| Return Features | 9 |
| Rolling Statistics | 8 |
| Lagged Features | 18 |
| Candlestick Features | 6 |
| Volume Features | 6 |
| Momentum & Oscillator | 11 |
| Trend Confirmation | 9 |
| **Tổng** | **~100+** |

### 2.4 Feature Scaling

Sử dụng **`RobustScaler`** (sklearn) thay vì `StandardScaler`:
- Ít nhạy cảm hơn với **outliers** (giá spike đột biến trên thị trường crypto)
- Scale dựa trên median và IQR thay vì mean/std

---

## 3. 🧠 Models (Mô hình AI)

### 3.1 Ensemble Architecture

Hệ thống kết hợp **3 mô hình song song** theo cơ chế **Weighted Voting** với trọng số thích ứng:

```
        Input Features (scaled)
               │
    ┌──────────┼────────────┐
    │          │            │
    ▼          ▼            ▼
XGBoost   Random Forest   LSTM
 (W_xgb)     (W_rf)     (W_lstm)
    │          │            │
    └──────────┴────────────┘
               │
       Weighted Average
        (ensemble_prob)
               │
       ┌───────┴───────┐
       │ Ngưỡng động   │
       │best_threshold │
       └───────┬───────┘
    ┌──────────┼──────────┐
    ▼          ▼          ▼
P > upper  P < lower    Else
  (MUA)      (BÁN)      (GIỮ)
```

*Trọng số thích ứng ($W_i$) được tính tự động từ giá trị $F1\text{-}\text{score}$ của từng mô hình trên tập Validation khi huấn luyện ($W_i = F1_i / \sum F1_j$). Trọng số mặc định nếu không có validation metrics là: XGBoost (25%), Random Forest (25%), LSTM (50%).*

### 3.2 XGBoost Classifier

| Tham số | Giá trị | Ghi chú |
|---------|---------|---------|
| Objective | `binary:logistic` | Phân loại nhị phân |
| `n_estimators` | 200–700 (search) | Số cây quyết định |
| `max_depth` | 3–9 (search) | Giới hạn chiều sâu cây |
| `learning_rate` | 0.005–0.05 (search) | Tốc độ học |
| `subsample` | 0.7–0.9 (search) | Tỷ lệ mẫu mỗi cây |
| `colsample_bytree` | 0.6–0.8 (search) | Tỷ lệ features mỗi cây |
| `scale_pos_weight` | `neg/pos` (tự tính) | Xử lý class imbalance |
| `eval_metric` | `logloss` | |
| Hyperparameter Tuning | `RandomizedSearchCV` + `TimeSeriesSplit(n_splits=5)` | |
| Scoring | `F1` | Cân bằng precision & recall |
| **Dữ liệu train** | `X_tree` (đã lọc sideway) | Chỉ train trên mẫu có `is_valid_target=1` |

### 3.3 Random Forest Classifier

| Tham số | Giá trị | Ghi chú |
|---------|---------|---------|
| `n_estimators` | 200–500 (search) | Số cây |
| `max_depth` | 5, 7, 10, 15, None (search) | Chiều sâu cây |
| `min_samples_split` | 5, 10, 20 (search) | Ngưỡng chia nhánh |
| `min_samples_leaf` | 3, 5, 10 (search) | Ngưỡng lá cây |
| `max_features` | `sqrt`, `log2` (search) | Số features mỗi split |
| `class_weight` | `balanced_subsample` | Xử lý class imbalance |
| Hyperparameter Tuning | `RandomizedSearchCV` + `TimeSeriesSplit(n_splits=5)` | |
| **Dữ liệu train** | `X_tree` (đã lọc sideway) | Chỉ train trên mẫu có `is_valid_target=1` |

### 3.4 LSTM (Long Short-Term Memory) with Attention

| Tham số | Giá trị | Ghi chú |
|---------|---------|---------|
| `sequence_length` | 48 nến | Ngữ cảnh 48 giờ trước |
| `hidden_size` | 128 | Kích thước hidden state |
| `num_layers` | 2 | Số tầng LSTM |
| `dropout` | 0.4 | Regularization |
| `learning_rate` | 0.0005 (Adam) | |
| `epochs` | 120 (max) | |
| `batch_size` | 64 | |
| `patience` | 20 | Early stopping |
| LR Scheduler | `ReduceLROnPlateau(factor=0.5, patience=8)` | Giảm LR khi loss plateau |
| Gradient Clipping | `max_norm=1.0` | Tránh gradient explosion |
| Kỹ thuật cải tiến | `Label Smoothing (0.1)` + `Class Weighting` | Cải thiện độ tin cậy và cân bằng mẫu |
| **Dữ liệu train** | `X_full` (toàn bộ chuỗi) | LSTM cần chuỗi liên tục để học temporal context |

**Kiến trúc LSTM with Attention:**
```
Input (seq_len=48, n_features)
    → LSTM Layer 1 (hidden=128, dropout=0.4)
    → LSTM Layer 2 (hidden=128, dropout=0.4)
    → Attention Layer (Tính toán động trọng số các timestep trong quá khứ)
    → Dropout(0.4)
    → Dense(128 → 64) + ReLU
    → Dropout(0.2)
    → Dense(64 → 32) + ReLU
    → Dense(32 → 1) + Sigmoid
Output (probability ∈ [0, 1])
```

### 3.5 Ensemble Prediction

```python
# Tính toán xác suất kết hợp bằng trung bình trọng số thích ứng
ensemble_prob = sum(p * w for p, w in zip(probs, weights_used)) / total_w

# best_threshold được tối ưu hóa động (Dynamic Thresholding) trên tập validation
# Tiêu chí tối ưu: F0.5-score (Ưu tiên Precision gấp 2x Recall để hạn chế tín hiệu giả)
thresh = best_threshold
margin = 0.10  # Vùng đệm GIỮ = [thresh - margin, thresh + margin]
upper = thresh + margin
lower = max(thresh - margin, 0.05)

if ensemble_prob > upper:
    signal = "MUA"
    # Độ tin cậy tính từ 33% (tại upper) đến 100% (tại 1.0)
    confidence = 33 + (ensemble_prob - upper) / (1.0 - upper) * 67
elif ensemble_prob < lower:
    signal = "BÁN"
    # Độ tin cậy tính từ 33% (tại lower) đến 100% (tại 0.0)
    confidence = 33 + (lower - ensemble_prob) / lower * 67
else:
    signal = "GIỮ"
    # Độ tin cậy cao nhất khi nằm chính giữa vùng đệm
    center = (upper + lower) / 2
    half_width = (upper - lower) / 2
    dist = abs(ensemble_prob - center) / half_width if half_width > 0 else 0
    confidence = 33 + (1 - dist) * 33  # 33% ở rìa biên, 66% ở giữa
```

---

## 4. 📊 Train/Validation Split

Hệ thống dùng **chronological split** (không shuffle) để tôn trọng tính thời gian:

```
Toàn bộ dữ liệu
├── 80% đầu (train)
└── 20% cuối (validation)
```

Metrics đánh giá: **Accuracy**, **F1-score**, **Precision**, **Recall**

---

## 5. 📡 Signal Generation Pipeline

```
OHLCV mới nhất
    │
    ▼
TechnicalIndicators.calculate_all()     → Tính indicators
    │
    ▼
FeatureEngineer.create_features()       → Tạo ML features
    │
    ▼
FeatureEngineer.get_latest_features()   → Scale features (RobustScaler)
    │
    ▼
EnsemblePredictor.predict()             → Dự đoán XGB + RF + LSTM
    │
    ▼
TradingStrategy._combine_signals()      → Kết hợp AI + Technical signals
    │
    ▼
Signal {MUA / BÁN / GIỮ, confidence%}
```

### Signal Combining (Kết hợp tín hiệu)

| Nguồn | Trọng số | Mô tả |
|-------|---------|-------|
| AI Ensemble | 55% | Xác suất từ mô hình |
| Momentum Score | 25% | Điểm tổng hợp technical |
| RSI | 10% | Tín hiệu quá mua/quá bán |
| MACD Histogram | 10% | Hướng momentum |

**High Confidence Bonus**: Khi AI và tất cả chỉ báo đồng thuận, confidence được nhân hệ số `×1.35` (tối đa 100%).

---

## 6. 🗂️ MLflow Tracking (Databricks)

Mỗi lần training được log đầy đủ lên Databricks MLflow:
- **Experiment**: `/Shared/AI_Trading_Experiments`
- **Metrics**: accuracy, f1, precision, recall (cho cả 3 models)
- **Params**: hyperparameters được chọn sau RandomizedSearchCV
- **Artifacts**: model files (xgb.pkl, rf.pkl, lstm.pth, scaler.pkl, meta.json, features.txt, feature_importance.json)

---

## 7. 💾 Model Storage

Mỗi model được lưu tại `models/{model_name}/`:

```
models/
└── BTC_USDT_{run_name}/
    ├── xgb.pkl        ← XGBoost (sklearn/joblib)
    ├── rf.pkl         ← Random Forest (sklearn/joblib)
    ├── lstm.pth       ← LSTM weights (PyTorch state_dict)
    ├── scaler.pkl     ← RobustScaler fitted
    ├── features.txt   ← Danh sách cột features
    ├── feature_importance.json ← Top features từ XGBoost
    └── meta.json      ← Training info (best_threshold, weights, metrics)
```

---

## 8. ⚙️ Configuration Nhanh

```python
# config.py (trích)

# Số nến tương lai để tính target
target_periods = 24         # 24 giờ tới (1 ngày trên khung 1H)

# Ngưỡng return tối thiểu (Dùng ATR Dynamic Threshold trong Databricks, config local chỉ fallback)
min_return_thresholds = {
    "BTC/USDT": 0.005,
    "ETH/USDT": 0.008,
    "default": 0.005,
}

# Ensemble weights (Đồng bộ với Databricks)
ensemble_weights = {
    "xgboost": 0.25,
    "random_forest": 0.25,
    "lstm": 0.50,
}

# Ngưỡng signal (Sẽ bị override bằng best_threshold từ meta.json)
confidence_threshold = 0.50
```
