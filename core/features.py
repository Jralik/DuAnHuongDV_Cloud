"""
Feature Engineering - Tạo features cho ML models
Cải thiện: fix target labels, thêm features mạnh, sửa lỗi streak calculation
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import RobustScaler  # dùng RobustScaler thay StandardScaler (ít nhạy với outliers)

import config


class FeatureEngineer:
    """Tạo và xử lý features cho AI/ML models"""

    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_columns: List[str] = []
        self._is_fitted = False

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo tất cả features từ dữ liệu OHLCV + indicators

        Args:
            df: DataFrame đã có indicators

        Returns:
            DataFrame với features bổ sung
        """
        if df.empty or len(df) < 60:
            return df

        result = df.copy()

        # === Price-based features ===
        result = self._price_features(result)

        # === Return-based features ===
        result = self._return_features(result)

        # === Rolling statistics ===
        result = self._rolling_features(result)

        # === Lagged features ===
        result = self._lagged_features(result)

        # === Candlestick patterns (đã fix lỗi streak) ===
        result = self._candle_features(result)

        # === Volume features ===
        result = self._volume_features(result)

        # === Momentum & Oscillator features (MỚI) ===
        result = self._momentum_features(result)

        # === Trend features (MỚI) ===
        result = self._trend_features(result)

        return result

    def _price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price-related features"""
        # Price relative to moving averages
        if "sma_20" in df.columns:
            df["price_sma20_ratio"] = df["close"] / df["sma_20"].replace(0, np.nan)
        if "sma_50" in df.columns:
            df["price_sma50_ratio"] = df["close"] / df["sma_50"].replace(0, np.nan)
        if "ema_21" in df.columns:
            df["price_ema21_ratio"] = df["close"] / df["ema_21"].replace(0, np.nan)

        # MA crossover (SMA10 vs SMA20)
        if "sma_10" in df.columns and "sma_20" in df.columns:
            df["sma10_sma20_cross"] = (df["sma_10"] - df["sma_20"]) / df["close"]

        # Price distance from high/low
        df["high_low_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
        df["close_open_range"] = (df["close"] - df["open"]) / df["close"].replace(0, np.nan)

        # Bollinger Band %B (vị trí giá trong dải BB) — feature mạnh
        if "bb_upper" in df.columns and "bb_lower" in df.columns:
            bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
            df["bb_pct_b"] = (df["close"] - df["bb_lower"]) / bb_range
            df["bb_width"] = bb_range / df["close"]

        return df

    def _return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return-based features"""
        # Simple returns
        df["return_1"] = df["close"].pct_change(1)
        df["return_3"] = df["close"].pct_change(3)
        df["return_6"] = df["close"].pct_change(6)
        df["return_12"] = df["close"].pct_change(12)
        df["return_24"] = df["close"].pct_change(24)

        # Log returns
        df["log_return_1"] = np.log(df["close"] / df["close"].shift(1))

        # Cumulative returns
        df["cum_return_6"] = df["return_1"].rolling(window=6).sum()
        df["cum_return_12"] = df["return_1"].rolling(window=12).sum()
        df["cum_return_24"] = df["return_1"].rolling(window=24).sum()

        return df

    def _rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling window statistics"""
        for window in [6, 12, 24]:
            # Volatility (realized)
            df[f"volatility_{window}"] = df["return_1"].rolling(window=window).std()

            # Z-score of price
            rolling_mean = df["close"].rolling(window=window).mean()
            rolling_std = df["close"].rolling(window=window).std().replace(0, np.nan)
            df[f"zscore_{window}"] = (df["close"] - rolling_mean) / rolling_std

        # Skewness chỉ tính với window đủ lớn (≥ 20)
        df["skew_20"] = df["return_1"].rolling(window=20).skew()
        df["kurt_20"] = df["return_1"].rolling(window=20).kurt()

        return df

    def _lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lagged values của các chỉ báo quan trọng"""
        lag_cols = ["rsi", "macd_histogram", "momentum_score", "adx"]
        for col in lag_cols:
            if col in df.columns:
                for lag in [1, 3, 6, 12]:
                    df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        # RSI change (tốc độ thay đổi RSI — phát hiện divergence)
        if "rsi" in df.columns:
            df["rsi_change_3"] = df["rsi"] - df["rsi"].shift(3)
            df["rsi_change_6"] = df["rsi"] - df["rsi"].shift(6)

        return df

    def _candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Candlestick-derived features (đã fix lỗi bullish/bearish streak)"""
        # Body size (relative)
        df["body_size"] = abs(df["close"] - df["open"]) / df["close"].replace(0, np.nan)

        # Upper/Lower shadows
        df["upper_shadow"] = (df["high"] - df[["close", "open"]].max(axis=1)) / df["close"].replace(0, np.nan)
        df["lower_shadow"] = (df[["close", "open"]].min(axis=1) - df["low"]) / df["close"].replace(0, np.nan)

        # Bullish/Bearish candle
        df["is_bullish"] = (df["close"] > df["open"]).astype(int)

        # FIX: Dùng cách tính streak đúng
        # Thuật toán: tích lũy đếm, reset khi đổi hướng
        is_bull = df["is_bullish"].values
        bull_streak = np.zeros(len(is_bull), dtype=int)
        bear_streak = np.zeros(len(is_bull), dtype=int)

        for i in range(1, len(is_bull)):
            if is_bull[i] == 1:
                bull_streak[i] = bull_streak[i - 1] + 1
                bear_streak[i] = 0
            else:
                bear_streak[i] = bear_streak[i - 1] + 1
                bull_streak[i] = 0

        df["bullish_streak"] = bull_streak
        df["bearish_streak"] = bear_streak

        return df

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based features"""
        # Volume relative to average (Z-score)
        vol_mean = df["volume"].rolling(window=20).mean()
        vol_std = df["volume"].rolling(window=20).std().replace(0, np.nan)
        df["volume_zscore"] = (df["volume"] - vol_mean) / vol_std

        # Volume ratio so với SMA20
        df["volume_sma20_ratio"] = df["volume"] / vol_mean.replace(0, np.nan)

        # Volume trend
        df["volume_change_3"] = df["volume"].pct_change(3)

        # On-Balance Volume momentum
        if "obv" in df.columns:
            df["obv_change_12"] = df["obv"].pct_change(12)

        # Volume x Price (Dollar Volume change)
        df["dollar_volume"] = df["close"] * df["volume"]
        df["dollar_volume_change"] = df["dollar_volume"].pct_change(6)

        return df

    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum & Oscillator features bổ sung — MỚI"""
        # RSI zones (one-hot encoding)
        if "rsi" in df.columns:
            df["rsi_oversold"] = (df["rsi"] < 30).astype(int)
            df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
            df["rsi_neutral"] = ((df["rsi"] >= 40) & (df["rsi"] <= 60)).astype(int)
            df["rsi_normalized"] = (df["rsi"] - 50) / 50  # -1 đến +1

        # Stochastic signal
        if "stoch_k" in df.columns and "stoch_d" in df.columns:
            df["stoch_diff"] = df["stoch_k"] - df["stoch_d"]
            df["stoch_oversold"] = (df["stoch_k"] < 20).astype(int)
            df["stoch_overbought"] = (df["stoch_k"] > 80).astype(int)

        # MACD signal strength
        if "macd" in df.columns and "macd_signal" in df.columns:
            df["macd_cross"] = np.sign(df["macd"] - df["macd_signal"])
            df["macd_cross_change"] = df["macd_cross"].diff()

        # MFI zones
        if "mfi" in df.columns:
            df["mfi_oversold"] = (df["mfi"] < 20).astype(int)
            df["mfi_overbought"] = (df["mfi"] > 80).astype(int)
            df["mfi_normalized"] = (df["mfi"] - 50) / 50

        return df

    def _trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend strength features — MỚI"""
        # ADX regime
        if "adx" in df.columns:
            df["adx_strong"] = (df["adx"] > 25).astype(int)
            df["adx_slope"] = df["adx"].diff(3)  # tốc độ tăng ADX

        # EMA alignment (bull/bear trend confirmation)
        if all(c in df.columns for c in ["ema_9", "ema_21"]):
            df["ema_alignment"] = np.sign(df["ema_9"] - df["ema_21"])
            df["ema_alignment_change"] = df["ema_alignment"].diff()

        # Higher highs / Lower lows (swing structure)
        df["higher_high_12"] = (df["high"] > df["high"].rolling(12).max().shift(1)).astype(int)
        df["lower_low_12"] = (df["low"] < df["low"].rolling(12).min().shift(1)).astype(int)

        # Price momentum (rate of change)
        df["roc_6"] = df["close"].pct_change(6)    # 6 giờ
        df["roc_12"] = df["close"].pct_change(12)  # 12 giờ
        df["roc_24"] = df["close"].pct_change(24)  # 24 giờ

        return df

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_periods: int = None,
        target_type: str = "direction",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Chuẩn bị dữ liệu cho training.

        QUAN TRỌNG: Dùng min_return_threshold để loại bỏ các mẫu sideway,
        giúp model chỉ học signal rõ ràng → accuracy cao hơn.

        Args:
            df: DataFrame with features
            target_periods: Số nến tương lai (mặc định từ config)
            target_type: 'direction' hoặc 'return'

        Returns:
            X (features), y (target)
        """
        n_periods = target_periods or config.MODEL_CONFIG.get("target_periods", 12)
        min_return = config.MODEL_CONFIG.get("min_return_threshold", 0.005)

        data = df.copy()

        # Tạo future return
        future_return = data["close"].shift(-n_periods) / data["close"] - 1

        if target_type == "direction":
            # FIX: Chỉ giữ các mẫu có return đủ lớn (lọc sideway noise)
            # Label 1 (MUA): giá tăng > min_return
            # Label 0 (BÁN): giá giảm > min_return
            # Bỏ qua các mẫu sideway |return| < min_return
            strong_up = future_return > min_return
            strong_down = future_return < -min_return
            mask = strong_up | strong_down

            data["target"] = np.where(strong_up, 1, 0)
            data = data[mask]  # Chỉ giữ signals rõ ràng
        else:
            data["target"] = future_return

        # Chọn feature columns (loại bỏ OHLCV gốc và target)
        exclude_cols = ["open", "high", "low", "close", "volume", "target",
                        "high_20", "low_20"]  # loại bỏ các cột không cần
        self.feature_columns = [
            col for col in data.columns
            if col not in exclude_cols
            and not any(col.startswith(p) for p in ["high_20", "low_20"])
        ]

        # Loại bỏ NaN
        data = data.dropna(subset=self.feature_columns + ["target"])
        data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=self.feature_columns)

        if data.empty or len(data) < 50:
            return pd.DataFrame(), pd.Series()

        X = data[self.feature_columns]
        y = data["target"].astype(int)

        return X, y

    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Chuẩn hóa features dùng RobustScaler (ít nhạy với outliers)"""
        if X.empty:
            return X

        if fit:
            scaled = self.scaler.fit_transform(X)
            self._is_fitted = True
        else:
            if not self._is_fitted:
                scaled = self.scaler.fit_transform(X)
                self._is_fitted = True
            else:
                scaled = self.scaler.transform(X)

        return pd.DataFrame(scaled, columns=X.columns, index=X.index)

    def get_latest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lấy features cho nến mới nhất (dùng cho prediction)"""
        if df.empty or not self.feature_columns:
            return pd.DataFrame()

        available_cols = [c for c in self.feature_columns if c in df.columns]
        latest = df[available_cols].iloc[[-1]].copy()
        latest = latest.fillna(0).replace([np.inf, -np.inf], 0)

        if self._is_fitted:
            # Đảm bảo đủ columns
            for col in self.feature_columns:
                if col not in latest.columns:
                    latest[col] = 0
            latest = latest[self.feature_columns]
            scaled = self.scaler.transform(latest)
            return pd.DataFrame(scaled, columns=self.feature_columns, index=latest.index)

        return latest
