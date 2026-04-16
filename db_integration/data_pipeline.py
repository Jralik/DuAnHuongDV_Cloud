"""
Data Pipeline - Đồng bộ dữ liệu Crypto lên Databricks Delta Lake
Hỗ trợ: OHLCV data, trade history, AI signals
"""
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

import config
from db_integration.client import DatabricksClient, get_client


class DataPipeline:
    """
    ETL Pipeline cho dữ liệu crypto trên Databricks.
    Sử dụng SQL Warehouse để tạo và quản lý Delta Lake tables.
    """

    def __init__(self, client: DatabricksClient = None):
        self.client = client or get_client()
        self.catalog = config.DATABRICKS_CONFIG["catalog"]
        self.schema = config.DATABRICKS_CONFIG["schema"]
        self._tables_ready = False

    @property
    def _full_schema(self) -> str:
        """Full schema path: catalog.schema"""
        return f"{self.catalog}.{self.schema}"

    def setup_tables(self) -> Dict:
        """
        Tạo schema và Delta tables trên Databricks.
        Idempotent: chạy nhiều lần không ảnh hưởng.
        """
        if not self.client:
            return {"success": False, "error": "Databricks chưa được kết nối"}

        results = {}

        try:
            # Tạo schema
            self.client.execute_statement(
                f"CREATE SCHEMA IF NOT EXISTS {self._full_schema}"
            )
            results["schema"] = "OK"

            # Table 1: OHLCV Data
            self.client.execute_statement(f"""
                CREATE TABLE IF NOT EXISTS {self._full_schema}.ohlcv_data (
                    timestamp STRING,
                    symbol STRING,
                    timeframe STRING,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume DOUBLE,
                    sync_time STRING
                )
            """)
            results["ohlcv_data"] = "OK"

            # Table 2: Trade History
            self.client.execute_statement(f"""
                CREATE TABLE IF NOT EXISTS {self._full_schema}.trade_history (
                    symbol STRING,
                    side STRING,
                    entry_price DOUBLE,
                    exit_price DOUBLE,
                    size DOUBLE,
                    pnl DOUBLE,
                    pnl_pct DOUBLE,
                    entry_time STRING,
                    exit_time STRING,
                    reason STRING,
                    stop_loss DOUBLE,
                    take_profit DOUBLE,
                    sync_time STRING
                )
            """)
            results["trade_history"] = "OK"

            # Table 3: AI Signals
            self.client.execute_statement(f"""
                CREATE TABLE IF NOT EXISTS {self._full_schema}.ai_signals (
                    timestamp STRING,
                    symbol STRING,
                    signal STRING,
                    confidence DOUBLE,
                    price DOUBLE,
                    reason STRING,
                    sync_time STRING
                )
            """)
            results["ai_signals"] = "OK"

            # Table 4: Training Results
            self.client.execute_statement(f"""
                CREATE TABLE IF NOT EXISTS {self._full_schema}.training_results (
                    timestamp STRING,
                    symbol STRING,
                    model_name STRING,
                    days INT,
                    data_points INT,
                    features_count INT,
                    xgb_accuracy DOUBLE,
                    xgb_f1 DOUBLE,
                    rf_accuracy DOUBLE,
                    rf_f1 DOUBLE,
                    lstm_accuracy DOUBLE,
                    lstm_f1 DOUBLE,
                    sync_time STRING
                )
            """)
            results["training_results"] = "OK"

            self._tables_ready = True
            return {"success": True, "tables": results}

        except Exception as e:
            return {"success": False, "error": str(e), "tables": results}

    def sync_ohlcv(self, symbol: str, df: pd.DataFrame,
                   timeframe: str = "1h") -> Dict:
        """
        Upload dữ liệu OHLCV lên Delta Lake.

        Args:
            symbol: Cặp giao dịch (VD: 'BTC/USDT')
            df: DataFrame OHLCV
            timeframe: Khung thời gian

        Returns:
            Dict kết quả đồng bộ
        """
        if not self.client:
            return {"success": False, "error": "Databricks chưa kết nối"}

        if df.empty:
            return {"success": False, "error": "DataFrame trống"}

        if not self._tables_ready:
            setup_result = self.setup_tables()
            if not setup_result.get("success"):
                return setup_result

        try:
            sync_time = datetime.now().isoformat()
            table = f"{self._full_schema}.ohlcv_data"

            # Xóa dữ liệu cũ của symbol + timeframe này
            self.client.execute_statement(
                f"DELETE FROM {table} WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'"
            )

            # Insert từng batch (tối đa 500 rows mỗi batch)
            rows_inserted = 0
            batch_size = 100
            values_list = []

            for idx, row in df.iterrows():
                ts = idx.isoformat() if hasattr(idx, 'isoformat') else str(idx)
                values_list.append(
                    f"('{ts}', '{symbol}', '{timeframe}', "
                    f"{row['open']}, {row['high']}, {row['low']}, "
                    f"{row['close']}, {row['volume']}, '{sync_time}')"
                )

                if len(values_list) >= batch_size:
                    values_str = ", ".join(values_list)
                    self.client.execute_statement(
                        f"INSERT INTO {table} VALUES {values_str}"
                    )
                    rows_inserted += len(values_list)
                    values_list = []

            # Insert remaining rows
            if values_list:
                values_str = ", ".join(values_list)
                self.client.execute_statement(
                    f"INSERT INTO {table} VALUES {values_str}"
                )
                rows_inserted += len(values_list)

            return {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "rows_inserted": rows_inserted,
                "sync_time": sync_time,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def sync_trades(self, trades: List[Dict]) -> Dict:
        """Upload lịch sử giao dịch lên Delta Lake"""
        if not self.client:
            return {"success": False, "error": "Databricks chưa kết nối"}

        if not trades:
            return {"success": False, "error": "Không có trades để sync"}

        if not self._tables_ready:
            self.setup_tables()

        try:
            sync_time = datetime.now().isoformat()
            table = f"{self._full_schema}.trade_history"
            rows_inserted = 0

            for trade in trades:
                self.client.execute_statement(f"""
                    INSERT INTO {table} VALUES (
                        '{trade.get("symbol", "")}',
                        '{trade.get("side", "")}',
                        {trade.get("entry_price", 0)},
                        {trade.get("exit_price", 0)},
                        {trade.get("size", 0)},
                        {trade.get("pnl", 0)},
                        {trade.get("pnl_pct", 0)},
                        '{trade.get("entry_time", "")}',
                        '{trade.get("exit_time", "")}',
                        '{trade.get("reason", "")}',
                        {trade.get("stop_loss", 0)},
                        {trade.get("take_profit", 0)},
                        '{sync_time}'
                    )
                """)
                rows_inserted += 1

            return {
                "success": True,
                "rows_inserted": rows_inserted,
                "sync_time": sync_time,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def sync_signals(self, signals: List[Dict]) -> Dict:
        """Upload tín hiệu AI lên Delta Lake"""
        if not self.client:
            return {"success": False, "error": "Databricks chưa kết nối"}

        if not signals:
            return {"success": False, "error": "Không có signals để sync"}

        if not self._tables_ready:
            self.setup_tables()

        try:
            sync_time = datetime.now().isoformat()
            table = f"{self._full_schema}.ai_signals"
            rows_inserted = 0

            for sig in signals:
                self.client.execute_statement(f"""
                    INSERT INTO {table} VALUES (
                        '{sig.get("timestamp", "")}',
                        '{sig.get("symbol", "")}',
                        '{sig.get("action", sig.get("signal", ""))}',
                        {sig.get("confidence", 0)},
                        {sig.get("price", 0)},
                        '{sig.get("reason", "").replace("'", "''")}',
                        '{sync_time}'
                    )
                """)
                rows_inserted += 1

            return {
                "success": True,
                "rows_inserted": rows_inserted,
                "sync_time": sync_time,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def sync_training_result(self, result: Dict) -> Dict:
        """Lưu kết quả training lên Delta Lake"""
        if not self.client:
            return {"success": False, "error": "Databricks chưa kết nối"}

        if not self._tables_ready:
            self.setup_tables()

        try:
            sync_time = datetime.now().isoformat()
            table = f"{self._full_schema}.training_results"
            metrics = result.get("metrics", {})

            xgb = metrics.get("xgboost", {})
            rf = metrics.get("random_forest", {})
            lstm = metrics.get("lstm", {})

            self.client.execute_statement(f"""
                INSERT INTO {table} VALUES (
                    '{result.get("timestamp", sync_time)}',
                    '{result.get("symbol", "")}',
                    '{result.get("model_name", "")}',
                    {result.get("days", 0)},
                    {result.get("data_points", 0)},
                    {result.get("features", 0)},
                    {xgb.get("accuracy", 0)},
                    {xgb.get("f1", 0)},
                    {rf.get("accuracy", 0)},
                    {rf.get("f1", 0)},
                    {lstm.get("accuracy", 0)},
                    {lstm.get("f1", 0)},
                    '{sync_time}'
                )
            """)

            return {"success": True, "sync_time": sync_time}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_ohlcv(self, symbol: str, timeframe: str = "1h",
                  limit: int = 500) -> pd.DataFrame:
        """Download dữ liệu OHLCV từ Delta Lake"""
        if not self.client:
            return pd.DataFrame()

        try:
            table = f"{self._full_schema}.ohlcv_data"
            rows = self.client.execute_query(f"""
                SELECT timestamp, open, high, low, close, volume
                FROM {table}
                WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
                ORDER BY timestamp DESC
                LIMIT {limit}
            """)

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            return df

        except Exception as e:
            print(f"[DATABRICKS] Lỗi get_ohlcv: {e}")
            return pd.DataFrame()

    def get_training_history(self, symbol: str = None,
                            limit: int = 20) -> List[Dict]:
        """Lấy lịch sử training từ Delta Lake"""
        if not self.client:
            return []

        try:
            table = f"{self._full_schema}.training_results"
            where = f"WHERE symbol = '{symbol}'" if symbol else ""
            rows = self.client.execute_query(f"""
                SELECT * FROM {table}
                {where}
                ORDER BY timestamp DESC
                LIMIT {limit}
            """)
            return rows

        except Exception as e:
            print(f"[DATABRICKS] Lỗi get_training_history: {e}")
            return []

    def get_sync_status(self) -> Dict:
        """Trạng thái đồng bộ các tables"""
        if not self.client:
            return {"enabled": False}

        status = {"enabled": True, "tables": {}}

        try:
            for table_name in ["ohlcv_data", "trade_history", "ai_signals", "training_results"]:
                try:
                    rows = self.client.execute_query(
                        f"SELECT COUNT(*) as cnt FROM {self._full_schema}.{table_name}"
                    )
                    status["tables"][table_name] = {
                        "exists": True,
                        "row_count": rows[0]["cnt"] if rows else 0,
                    }
                except Exception:
                    status["tables"][table_name] = {"exists": False, "row_count": 0}

        except Exception as e:
            status["error"] = str(e)

        return status


# Singleton instance
_pipeline_instance: Optional[DataPipeline] = None


def get_pipeline() -> Optional[DataPipeline]:
    """Lấy singleton DataPipeline instance"""
    global _pipeline_instance
    if not config.DATABRICKS_CONFIG.get("enabled"):
        return None
    if _pipeline_instance is None:
        _pipeline_instance = DataPipeline()
    return _pipeline_instance
