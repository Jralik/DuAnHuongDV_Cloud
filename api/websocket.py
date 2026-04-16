"""
WebSocket Manager - Real-time data streaming
"""
from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict
import json
import asyncio
from datetime import datetime


class WebSocketManager:
    """Quản lý WebSocket connections cho real-time data"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Chấp nhận kết nối mới"""
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WS] Client kết nối. Tổng: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Ngắt kết nối"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"[WS] Client ngắt kết nối. Tổng: {len(self.active_connections)}")

    async def broadcast(self, data: Dict):
        """Gửi dữ liệu đến tất cả clients"""
        if not self.active_connections:
            return

        message = json.dumps(data, default=str)
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)

    async def send_to(self, websocket: WebSocket, data: Dict):
        """Gửi dữ liệu đến một client cụ thể"""
        try:
            await websocket.send_text(json.dumps(data, default=str))
        except Exception:
            self.disconnect(websocket)


ws_manager = WebSocketManager()


async def start_price_stream(trading_engine):
    """Background task - stream giá real-time"""
    while True:
        try:
            if ws_manager.active_connections and trading_engine:
                prices = trading_engine.data_manager.get_latest_prices()

                # Update portfolio positions
                price_map = {
                    symbol: info.get("last", 0)
                    for symbol, info in prices.items()
                    if info.get("last", 0) > 0
                }
                trading_engine.portfolio.update_prices(price_map)

                await ws_manager.broadcast({
                    "type": "price_update",
                    "data": prices,
                    "portfolio": trading_engine.portfolio.get_summary(),
                    "timestamp": datetime.now().isoformat(),
                })

        except Exception as e:
            print(f"[WS] Lỗi stream: {e}")

        await asyncio.sleep(10)  # Cập nhật mỗi 10 giây


async def start_signal_stream(trading_engine):
    """Background task - stream tín hiệu real-time"""
    while True:
        try:
            if ws_manager.active_connections and trading_engine:
                signals = {}
                for symbol in trading_engine.symbols:
                    df = trading_engine.data_manager.get_data(symbol, "1h", 200)
                    if not df.empty:
                        signal = trading_engine.strategy.generate_signal(symbol, df)
                        signals[symbol] = signal.to_dict()

                if signals:
                    await ws_manager.broadcast({
                        "type": "signal_update",
                        "data": signals,
                        "timestamp": datetime.now().isoformat(),
                    })

        except Exception as e:
            print(f"[WS] Lỗi signal stream: {e}")

        await asyncio.sleep(60)  # Cập nhật mỗi 60 giây
