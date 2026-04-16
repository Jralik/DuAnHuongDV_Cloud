"""
Portfolio Manager - Quản lý danh mục & Paper Trading
"""
import json
import os
from typing import Dict, List, Optional
from datetime import datetime

import config
from core.risk_manager import RiskManager


class Position:
    """Đại diện một vị thế đang mở"""

    def __init__(self, symbol: str, side: str, entry_price: float,
                 size: float, stop_loss: float = 0, take_profit: float = 0):
        self.id = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.size = size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.entry_time = datetime.now()
        self.current_price = entry_price
        self.unrealized_pnl = 0

    def update_price(self, price: float):
        """Cập nhật giá hiện tại"""
        self.current_price = price
        if self.side == "buy":
            self.unrealized_pnl = (price - self.entry_price) * self.size
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.size

    def check_stop_loss(self, price: float) -> bool:
        """Kiểm tra stop loss"""
        if self.stop_loss <= 0:
            return False
        if self.side == "buy" and price <= self.stop_loss:
            return True
        if self.side == "sell" and price >= self.stop_loss:
            return True
        return False

    def check_take_profit(self, price: float) -> bool:
        """Kiểm tra take profit"""
        if self.take_profit <= 0:
            return False
        if self.side == "buy" and price >= self.take_profit:
            return True
        if self.side == "sell" and price <= self.take_profit:
            return True
        return False

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": round(self.entry_price, 2),
            "current_price": round(self.current_price, 2),
            "size": round(self.size, 6),
            "value": round(self.size * self.current_price, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(
                self.unrealized_pnl / (self.entry_price * self.size) * 100, 2
            ) if self.entry_price * self.size > 0 else 0,
            "stop_loss": round(self.stop_loss, 2),
            "take_profit": round(self.take_profit, 2),
            "entry_time": self.entry_time.isoformat(),
        }


class TradeRecord:
    """Lịch sử giao dịch đã đóng"""

    def __init__(self, position: Position, exit_price: float, reason: str):
        self.symbol = position.symbol
        self.side = position.side
        self.entry_price = position.entry_price
        self.exit_price = exit_price
        self.size = position.size
        self.entry_time = position.entry_time
        self.exit_time = datetime.now()
        self.reason = reason
        self.stop_loss = position.stop_loss
        self.take_profit = position.take_profit

        if position.side == "buy":
            self.pnl = (exit_price - position.entry_price) * position.size
        else:
            self.pnl = (position.entry_price - exit_price) * position.size

        # Trừ phí
        commission = (position.entry_price + exit_price) * position.size * config.BACKTEST_CONFIG["commission"]
        self.pnl -= commission

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(self.exit_price, 2),
            "size": round(self.size, 6),
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(
                self.pnl / (self.entry_price * self.size) * 100, 2
            ) if self.entry_price * self.size > 0 else 0,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "reason": self.reason,
            "stop_loss": round(self.stop_loss, 2),
            "take_profit": round(self.take_profit, 2),
        }


class Portfolio:
    """
    Paper Trading Portfolio Manager
    Quản lý vốn, vị thế, và lịch sử giao dịch
    """

    def __init__(self, initial_capital: float = None):
        self.initial_capital = initial_capital or config.INITIAL_CAPITAL
        self.cash = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[TradeRecord] = []
        self.equity_history: List[Dict] = []
        self.risk_manager = RiskManager(self.initial_capital)
        self.peak_equity = self.initial_capital  # Track highest equity for drawdown

        # Record initial state
        self._record_equity()

    @property
    def total_equity(self) -> float:
        """Tổng giá trị portfolio (cash + position values)"""
        position_value = sum(p.size * p.current_price for p in self.positions.values())
        return self.cash + position_value

    @property
    def total_pnl(self) -> float:
        """Tổng P&L"""
        return self.total_equity - self.initial_capital

    @property
    def total_pnl_pct(self) -> float:
        """Tổng P&L %"""
        return (self.total_pnl / self.initial_capital) * 100

    def open_position(self, symbol: str, side: str, price: float,
                      size: float = None, atr: float = None) -> Dict:
        """
        Mở vị thế mới

        Args:
            symbol: Symbol
            side: "buy" hoặc "sell"
            price: Giá hiện tại
            size: Kích thước (nếu None, tự tính theo risk)
            atr: ATR value cho SL/TP

        Returns:
            Dict kết quả
        """
        # Check risk limits — dùng peak_equity để tính drawdown chính xác
        risk_check = self.risk_manager.check_risk_limits(
            self.total_equity,
            len(self.positions),
            self._get_daily_pnl(),
            self.peak_equity,
        )

        if not risk_check["can_trade"]:
            return {
                "success": False,
                "error": "Không thể mở lệnh: " + ", ".join(risk_check["warnings"]),
                "warnings": risk_check["warnings"],
            }

        # Check nếu đã có position cho symbol này
        if symbol in self.positions:
            return {
                "success": False,
                "error": f"Đã có vị thế mở cho {symbol}. Vui lòng đóng vị thế cũ trước khi mở lệnh mới.",
            }

        # Calculate SL/TP
        sl_tp = self.risk_manager.calculate_stop_loss(price, side, atr)

        # Calculate position size
        if size is None:
            pos_info = self.risk_manager.calculate_position_size(
                self.total_equity, price, sl_tp["stop_loss"]
            )
            size = pos_info["size"]

        # Check đủ vốn
        required = price * size
        if required > self.cash:
            size = self.cash * 0.95 / price  # Dùng 95% vốn khả dụng

        # Open position
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=price,
            size=size,
            stop_loss=sl_tp["stop_loss"],
            take_profit=sl_tp["take_profit"],
        )

        self.positions[symbol] = position
        self.cash -= price * size

        self._record_equity()

        return {
            "success": True,
            "position": position.to_dict(),
            "remaining_cash": round(self.cash, 2),
        }

    def close_position(self, symbol: str, price: float,
                       reason: str = "Thủ công") -> Dict:
        """Đóng vị thế"""
        if symbol not in self.positions:
            return {"success": False, "error": f"Không tìm thấy vị thế {symbol}"}

        position = self.positions[symbol]
        trade = TradeRecord(position, price, reason)
        self.trade_history.append(trade)

        # Trả lại vốn gốc + lãi/lỗ (trade.pnl đã tính sẵn chênh lệch giá)
        self.cash += position.entry_price * position.size + trade.pnl
        del self.positions[symbol]

        self._record_equity()

        # Tự động đẩy lịch sử lệnh này lên Databricks Cloud
        if config.DATABRICKS_CONFIG.get("enabled"):
            try:
                from db_integration.data_pipeline import get_pipeline
                import threading
                pipeline = get_pipeline()
                if pipeline:
                    threading.Thread(
                        target=pipeline.sync_trades,
                        args=([trade.to_dict()],),
                        daemon=True
                    ).start()
            except Exception as e:
                print(f"[DATABRICKS] Lỗi auto-sync trade: {e}")

        return {
            "success": True,
            "trade": trade.to_dict(),
            "remaining_cash": round(self.cash, 2),
        }

    def update_prices(self, prices: Dict[str, float]):
        """Cập nhật giá mới nhất cho tất cả positions"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos.update_price(price)

                # Auto SL/TP check
                if pos.check_stop_loss(price):
                    self.close_position(symbol, price, "Stop Loss")
                elif pos.check_take_profit(price):
                    self.close_position(symbol, price, "Take Profit")

    def _get_daily_pnl(self) -> float:
        """Tổng PnL trong ngày"""
        today = datetime.now().date()
        daily_pnl = sum(
            t.pnl for t in self.trade_history
            if t.exit_time.date() == today
        )
        return daily_pnl

    def _record_equity(self):
        """Ghi lại giá trị equity và cập nhật peak"""
        equity = self.total_equity
        # Cập nhật peak equity để tính drawdown chính xác
        if equity > self.peak_equity:
            self.peak_equity = equity
        self.equity_history.append({
            "timestamp": datetime.now().isoformat(),
            "equity": round(equity, 2),
            "cash": round(self.cash, 2),
            "positions_value": round(
                sum(p.size * p.current_price for p in self.positions.values()), 2
            ),
        })

    def get_summary(self) -> Dict:
        """Tổng quan portfolio"""
        realized_pnl = sum(t.pnl for t in self.trade_history)
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())

        return {
            "initial_capital": round(self.initial_capital, 2),
            "total_equity": round(self.total_equity, 2),
            "cash": round(self.cash, 2),
            "realized_pnl": round(realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "total_pnl": round(self.total_pnl, 2),
            "total_pnl_pct": round(self.total_pnl_pct, 2),
            "open_positions": len(self.positions),
            "total_trades": len(self.trade_history),
            "positions": [p.to_dict() for p in self.positions.values()],
            "recent_trades": [t.to_dict() for t in self.trade_history[-10:]],
            "equity_history": self.equity_history[-100:],
        }

    def save_state(self):
        """Lưu trạng thái portfolio"""
        state = {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "trade_history": [t.to_dict() for t in self.trade_history],
            "equity_history": self.equity_history,
        }
        filepath = os.path.join(config.DATA_DIR, "portfolio_state.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_state(self):
        """Đọc trạng thái portfolio"""
        filepath = os.path.join(config.DATA_DIR, "portfolio_state.json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.initial_capital = state.get("initial_capital", config.INITIAL_CAPITAL)
            self.cash = state.get("cash", self.initial_capital)
            self.equity_history = state.get("equity_history", [])
