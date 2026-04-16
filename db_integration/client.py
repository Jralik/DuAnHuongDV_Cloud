"""
Databricks Client - Quản lý kết nối Databricks SQL Warehouse
Sử dụng databricks-sql-connector để kết nối Delta Lake
"""
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

import config


class DatabricksClient:
    """
    Quản lý kết nối tới Databricks SQL Warehouse.
    Hỗ trợ: execute SQL queries, test connection, get status.
    """

    def __init__(self):
        self.host = config.DATABRICKS_CONFIG["host"]
        self.token = config.DATABRICKS_CONFIG["token"]
        self.http_path = config.DATABRICKS_CONFIG["http_path"]
        self.catalog = config.DATABRICKS_CONFIG["catalog"]
        self.schema = config.DATABRICKS_CONFIG["schema"]
        self._connection = None
        self._connected = False
        self._last_error = ""

    def _get_connection(self):
        """Tạo hoặc tái sử dụng connection"""
        if self._connection is not None:
            try:
                # Test if connection still alive
                cursor = self._connection.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                return self._connection
            except Exception:
                self._connection = None
                self._connected = False

        try:
            from databricks import sql as databricks_sql
            self._connection = databricks_sql.connect(
                server_hostname=self.host,
                http_path=self.http_path,
                access_token=self.token,
            )
            self._connected = True
            self._last_error = ""
            return self._connection
        except Exception as e:
            self._connected = False
            self._last_error = str(e)
            print(f"[DATABRICKS] Connection error: {e}")
            return None

    def test_connection(self) -> Dict:
        """Kiểm tra kết nối tới Databricks SQL Warehouse"""
        try:
            conn = self._get_connection()
            if conn is None:
                return {
                    "connected": False,
                    "error": self._last_error or "Không thể kết nối",
                }

            cursor = conn.cursor()
            cursor.execute("SELECT current_timestamp() as ts, current_user() as user")
            row = cursor.fetchone()
            cursor.close()

            return {
                "connected": True,
                "host": self.host,
                "timestamp": str(row[0]) if row else "",
                "user": str(row[1]) if row else "",
                "catalog": self.catalog,
                "schema": self.schema,
            }
        except Exception as e:
            self._connected = False
            self._last_error = str(e)
            return {"connected": False, "error": str(e)}

    def execute_query(self, sql: str, params: tuple = None) -> List[Dict]:
        """
        Chạy SQL query và trả về kết quả dạng list of dicts.

        Args:
            sql: SQL statement
            params: Optional query parameters

        Returns:
            List[Dict] kết quả
        """
        conn = self._get_connection()
        if conn is None:
            raise ConnectionError(f"Không thể kết nối Databricks: {self._last_error}")

        cursor = conn.cursor()
        try:
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)

            # Nếu là SELECT, trả về data
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                return [dict(zip(columns, row)) for row in rows]
            return []
        finally:
            cursor.close()

    def execute_statement(self, sql: str) -> bool:
        """Chạy SQL statement (CREATE, INSERT, DROP...) không trả data"""
        conn = self._get_connection()
        if conn is None:
            raise ConnectionError(f"Không thể kết nối Databricks: {self._last_error}")

        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            return True
        except Exception as e:
            print(f"[DATABRICKS] SQL Error: {e}")
            raise
        finally:
            cursor.close()

    def get_status(self) -> Dict:
        """Trạng thái kết nối hiện tại"""
        return {
            "enabled": config.DATABRICKS_CONFIG["enabled"],
            "connected": self._connected,
            "host": self.host,
            "catalog": self.catalog,
            "schema": self.schema,
            "last_error": self._last_error,
            "timestamp": datetime.now().isoformat(),
        }

    def close(self):
        """Đóng connection"""
        if self._connection:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None
            self._connected = False


# Singleton instance
_client_instance: Optional[DatabricksClient] = None


def get_client() -> Optional[DatabricksClient]:
    """Lấy singleton DatabricksClient instance"""
    global _client_instance
    if not config.DATABRICKS_CONFIG.get("enabled"):
        return None
    if _client_instance is None:
        _client_instance = DatabricksClient()
    return _client_instance
