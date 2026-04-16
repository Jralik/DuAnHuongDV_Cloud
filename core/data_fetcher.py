"""
Data Fetcher - Thu thập dữ liệu thị trường Crypto
Sử dụng CCXT để kết nối các sàn giao dịch
"""
import ccxt
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import asyncio

import config


class CryptoDataFetcher:
    """Thu thập dữ liệu crypto từ các sàn giao dịch qua CCXT"""

    def __init__(self, exchange_id: str = None):
        self.exchange_id = exchange_id or config.EXCHANGE_ID
        self.exchange = self._init_exchange()
        self._cache: Dict[str, pd.DataFrame] = {}

    def _init_exchange(self) -> ccxt.Exchange:
        """Khởi tạo kết nối sàn giao dịch"""
        exchange_class = getattr(ccxt, self.exchange_id)
        exchange = exchange_class({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        return exchange

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
        since: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Lấy dữ liệu OHLCV (Open, High, Low, Close, Volume)

        Args:
            symbol: Cặp giao dịch (VD: 'BTC/USDT')
            timeframe: Khung thời gian ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Số nến tối đa
            since: Timestamp bắt đầu (ms)

        Returns:
            DataFrame với columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, limit=limit, since=since
            )

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            # Convert to float
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            # Cache data
            cache_key = f"{symbol}_{timeframe}"
            self._cache[cache_key] = df

            return df

        except Exception as e:
            print(f"[ERROR] Lỗi khi lấy dữ liệu {symbol}: {e}")
            # Thử trả về cache nếu có
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self._cache:
                print(f"[INFO] Sử dụng dữ liệu cache cho {symbol}")
                return self._cache[cache_key]
            return pd.DataFrame()

    def fetch_ticker(self, symbol: str) -> Dict:
        """Lấy thông tin giá hiện tại"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                "symbol": symbol,
                "last": ticker.get("last", 0),
                "bid": ticker.get("bid", 0),
                "ask": ticker.get("ask", 0),
                "high": ticker.get("high", 0),
                "low": ticker.get("low", 0),
                "volume": ticker.get("baseVolume", 0),
                "change": ticker.get("percentage", 0),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            print(f"[ERROR] Lỗi khi lấy ticker {symbol}: {e}")
            return {
                "symbol": symbol,
                "last": 0, "bid": 0, "ask": 0,
                "high": 0, "low": 0, "volume": 0,
                "change": 0, "timestamp": datetime.now().isoformat(),
            }

    def fetch_multiple_symbols(
        self,
        symbols: List[str] = None,
        timeframe: str = "1h",
        limit: int = 500,
    ) -> Dict[str, pd.DataFrame]:
        """Lấy dữ liệu cho nhiều symbol"""
        symbols = symbols or config.SYMBOLS
        result = {}
        for symbol in symbols:
            df = self.fetch_ohlcv(symbol, timeframe, limit)
            if not df.empty:
                result[symbol] = df
        return result

    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str = "1h",
        days: int = 90,
    ) -> pd.DataFrame:
        """
        Lấy dữ liệu lịch sử dài hạn (phân trang)

        Args:
            symbol: Cặp giao dịch
            timeframe: Khung thời gian
            days: Số ngày lịch sử cần lấy

        Returns:
            DataFrame dữ liệu lịch sử
        """
        timeframe_ms = {
            "1m": 60000, "5m": 300000, "15m": 900000,
            "1h": 3600000, "4h": 14400000, "1d": 86400000,
        }

        tf_ms = timeframe_ms.get(timeframe, 3600000)
        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        all_data = []
        batch_limit = 500

        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, since=since, limit=batch_limit
                )
                if not ohlcv:
                    break

                all_data.extend(ohlcv)
                since = ohlcv[-1][0] + tf_ms

                if len(ohlcv) < batch_limit:
                    break

                # Rate limit
                import time
                time.sleep(self.exchange.rateLimit / 1000)

            except Exception as e:
                print(f"[ERROR] Lỗi khi lấy dữ liệu lịch sử: {e}")
                break

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(
            all_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        return df

    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Lưu dữ liệu ra file CSV"""
        filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
        filepath = os.path.join(config.DATA_DIR, filename)
        df.to_csv(filepath)
        print(f"[INFO] Đã lưu dữ liệu: {filepath}")

    def load_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Đọc dữ liệu từ file CSV"""
        filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
        filepath = os.path.join(config.DATA_DIR, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col="timestamp", parse_dates=True)
            return df
        return pd.DataFrame()

    def get_available_symbols(self) -> List[str]:
        """Lấy danh sách symbols có sẵn trên sàn"""
        try:
            self.exchange.load_markets()
            return list(self.exchange.markets.keys())
        except Exception as e:
            print(f"[ERROR] Lỗi khi lấy danh sách symbols: {e}")
            return []


class DataManager:
    """Quản lý dữ liệu tổng hợp"""

    def __init__(self):
        self.fetcher = CryptoDataFetcher()
        self.data_store: Dict[str, pd.DataFrame] = {}

    def get_data(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Lấy dữ liệu với caching thông minh"""
        cache_key = f"{symbol}_{timeframe}"

        if use_cache and cache_key in self.data_store:
            cached = self.data_store[cache_key]
            # Kiểm tra dữ liệu còn mới không (< 5 phút)
            if not cached.empty:
                last_time = cached.index[-1]
                if (datetime.now() - last_time.to_pydatetime().replace(tzinfo=None)).total_seconds() < 300:
                    return cached

        # Fetch dữ liệu mới
        df = self.fetcher.fetch_ohlcv(symbol, timeframe, limit)
        if not df.empty:
            self.data_store[cache_key] = df
        return df

    def refresh_all(self, symbols: List[str] = None, timeframe: str = "1h"):
        """Cập nhật dữ liệu cho tất cả symbols"""
        symbols = symbols or config.SYMBOLS
        for symbol in symbols:
            self.get_data(symbol, timeframe, use_cache=False)

    def get_latest_prices(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """Lấy giá mới nhất cho tất cả symbols"""
        symbols = symbols or config.SYMBOLS
        prices = {}
        for symbol in symbols:
            prices[symbol] = self.fetcher.fetch_ticker(symbol)
        return prices
