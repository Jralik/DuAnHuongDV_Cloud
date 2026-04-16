"""
Technical Indicators - Tính toán chỉ báo kỹ thuật
Sử dụng pandas-ta cho hiệu suất cao
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, Optional

import config


class TechnicalIndicators:
    """Tính toán các chỉ báo kỹ thuật trên dữ liệu OHLCV"""

    def __init__(self, indicator_config: Dict = None):
        self.config = indicator_config or config.INDICATOR_CONFIG

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính tất cả các chỉ báo kỹ thuật

        Args:
            df: DataFrame OHLCV

        Returns:
            DataFrame với các cột chỉ báo bổ sung
        """
        if df.empty or len(df) < 50:
            return df

        result = df.copy()

        # === TREND Indicators ===
        result = self._add_moving_averages(result)
        result = self._add_macd(result)
        result = self._add_adx(result)

        # === MOMENTUM Indicators ===
        result = self._add_rsi(result)
        result = self._add_stochastic(result)
        result = self._add_cci(result)
        result = self._add_williams_r(result)

        # === VOLATILITY Indicators ===
        result = self._add_bollinger_bands(result)
        result = self._add_atr(result)

        # === VOLUME Indicators ===
        result = self._add_obv(result)
        result = self._add_mfi(result)

        # === CUSTOM Indicators ===
        result = self._add_price_channels(result)
        result = self._add_momentum_score(result)

        return result

    # ------ TREND ------

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """SMA và EMA"""
        for period in self.config["sma_periods"]:
            df[f"sma_{period}"] = ta.sma(df["close"], length=period)

        for period in self.config["ema_periods"]:
            df[f"ema_{period}"] = ta.ema(df["close"], length=period)

        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD (Moving Average Convergence Divergence)"""
        macd_config = self.config["macd"]
        macd = ta.macd(
            df["close"],
            fast=macd_config["fast"],
            slow=macd_config["slow"],
            signal=macd_config["signal"],
        )
        if macd is not None and not macd.empty:
            df["macd"] = macd.iloc[:, 0]
            df["macd_histogram"] = macd.iloc[:, 1]
            df["macd_signal"] = macd.iloc[:, 2]
        return df

    def _add_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """ADX (Average Directional Index)"""
        adx = ta.adx(df["high"], df["low"], df["close"],
                      length=self.config["adx_period"])
        if adx is not None and not adx.empty:
            df["adx"] = adx.iloc[:, 0]
            df["di_plus"] = adx.iloc[:, 1]
            df["di_minus"] = adx.iloc[:, 2]
        return df

    # ------ MOMENTUM ------

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI (Relative Strength Index)"""
        df["rsi"] = ta.rsi(df["close"], length=self.config["rsi_period"])
        return df

    def _add_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stochastic Oscillator"""
        stoch_config = self.config["stoch"]
        stoch = ta.stoch(
            df["high"], df["low"], df["close"],
            k=stoch_config["k"], d=stoch_config["d"],
        )
        if stoch is not None and not stoch.empty:
            df["stoch_k"] = stoch.iloc[:, 0]
            df["stoch_d"] = stoch.iloc[:, 1]
        return df

    def _add_cci(self, df: pd.DataFrame) -> pd.DataFrame:
        """CCI (Commodity Channel Index)"""
        df["cci"] = ta.cci(
            df["high"], df["low"], df["close"],
            length=self.config["cci_period"],
        )
        return df

    def _add_williams_r(self, df: pd.DataFrame) -> pd.DataFrame:
        """Williams %R"""
        df["willr"] = ta.willr(
            df["high"], df["low"], df["close"],
            length=self.config["willr_period"],
        )
        return df

    # ------ VOLATILITY ------

    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands"""
        bb_config = self.config["bb"]
        bbands = ta.bbands(
            df["close"],
            length=bb_config["period"],
            std=bb_config["std"],
        )
        if bbands is not None and not bbands.empty:
            df["bb_lower"] = bbands.iloc[:, 0]
            df["bb_mid"] = bbands.iloc[:, 1]
            df["bb_upper"] = bbands.iloc[:, 2]
            df["bb_bandwidth"] = bbands.iloc[:, 3]
            df["bb_percent"] = bbands.iloc[:, 4]
        return df

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR (Average True Range)"""
        df["atr"] = ta.atr(
            df["high"], df["low"], df["close"],
            length=self.config["atr_period"],
        )
        return df

    # ------ VOLUME ------

    def _add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """OBV (On-Balance Volume)"""
        df["obv"] = ta.obv(df["close"], df["volume"])
        return df

    def _add_mfi(self, df: pd.DataFrame) -> pd.DataFrame:
        """MFI (Money Flow Index)"""
        df["mfi"] = ta.mfi(
            df["high"], df["low"], df["close"], df["volume"],
            length=self.config["mfi_period"],
        )
        return df

    # ------ CUSTOM ------

    def _add_price_channels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price channels và support/resistance"""
        df["high_20"] = df["high"].rolling(window=20).max()
        df["low_20"] = df["low"].rolling(window=20).min()
        df["mid_20"] = (df["high_20"] + df["low_20"]) / 2

        # Price position within channel (0 = bottom, 1 = top)
        channel_range = df["high_20"] - df["low_20"]
        df["price_position"] = np.where(
            channel_range > 0,
            (df["close"] - df["low_20"]) / channel_range,
            0.5
        )
        return df

    def _add_momentum_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Điểm momentum tổng hợp (-100 đến +100)
        Kết hợp nhiều chỉ báo momentum
        """
        score = pd.Series(0, index=df.index, dtype=float)
        weight_total = 0

        # RSI contribution
        if "rsi" in df.columns:
            rsi_score = (df["rsi"] - 50) * 2  # -100 to +100
            score += rsi_score * 0.3
            weight_total += 0.3

        # MACD contribution
        if "macd_histogram" in df.columns:
            macd_norm = df["macd_histogram"] / (df["close"] * 0.01)
            macd_score = macd_norm.clip(-100, 100)
            score += macd_score * 0.25
            weight_total += 0.25

        # Stochastic contribution
        if "stoch_k" in df.columns:
            stoch_score = (df["stoch_k"] - 50) * 2
            score += stoch_score * 0.2
            weight_total += 0.2

        # CCI contribution
        if "cci" in df.columns:
            cci_score = (df["cci"] / 2).clip(-100, 100)
            score += cci_score * 0.15
            weight_total += 0.15

        # ADX trend strength
        if "adx" in df.columns and "di_plus" in df.columns:
            trend_dir = np.where(df["di_plus"] > df["di_minus"], 1, -1)
            adx_score = df["adx"] * trend_dir
            score += adx_score * 0.1
            weight_total += 0.1

        if weight_total > 0:
            df["momentum_score"] = (score / weight_total).clip(-100, 100)
        else:
            df["momentum_score"] = 0

        return df

    def get_latest_signals(self, df: pd.DataFrame) -> Dict:
        """
        Lấy tín hiệu từ indicators ở nến cuối cùng

        Returns:
            Dict chứa giá trị các indicator và tín hiệu
        """
        if df.empty:
            return {}

        last = df.iloc[-1]
        signals = {
            "price": last.get("close", 0),
            "rsi": round(last.get("rsi", 50), 2),
            "macd": round(last.get("macd", 0), 4),
            "macd_signal": round(last.get("macd_signal", 0), 4),
            "macd_histogram": round(last.get("macd_histogram", 0), 4),
            "bb_upper": round(last.get("bb_upper", 0), 2),
            "bb_lower": round(last.get("bb_lower", 0), 2),
            "atr": round(last.get("atr", 0), 4),
            "adx": round(last.get("adx", 0), 2),
            "momentum_score": round(last.get("momentum_score", 0), 2),
            "stoch_k": round(last.get("stoch_k", 50), 2),
            "mfi": round(last.get("mfi", 50), 2),
        }

        # Phân tích tín hiệu
        signal_analysis = []

        # RSI
        rsi = signals["rsi"]
        if rsi < 30:
            signal_analysis.append(("RSI", "QUÁ BÁN", "bullish"))
        elif rsi > 70:
            signal_analysis.append(("RSI", "QUÁ MUA", "bearish"))
        else:
            signal_analysis.append(("RSI", "TRUNG LẬP", "neutral"))

        # MACD
        if signals["macd_histogram"] > 0:
            signal_analysis.append(("MACD", "TĂNG", "bullish"))
        else:
            signal_analysis.append(("MACD", "GIẢM", "bearish"))

        # Bollinger
        price = signals["price"]
        if price < signals["bb_lower"]:
            signal_analysis.append(("BB", "DƯỚI DẢI", "bullish"))
        elif price > signals["bb_upper"]:
            signal_analysis.append(("BB", "TRÊN DẢI", "bearish"))
        else:
            signal_analysis.append(("BB", "TRONG DẢI", "neutral"))

        # ADX Trend
        if signals["adx"] > 25:
            signal_analysis.append(("ADX", "XU HƯỚNG MẠNH", "strong"))
        else:
            signal_analysis.append(("ADX", "SIDEWAY", "weak"))

        signals["analysis"] = signal_analysis

        return signals
