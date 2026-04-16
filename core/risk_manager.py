"""
Risk Manager - Quản lý rủi ro giao dịch
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime, timedelta

import config


class RiskManager:
    """
    Quản lý rủi ro giao dịch
    - Position sizing
    - Stop-loss / Take-profit
    - Drawdown protection
    - Daily loss limits
    """

    def __init__(self, initial_capital: float = None):
        self.initial_capital = initial_capital or config.INITIAL_CAPITAL
        self.config = config.RISK_CONFIG
        self.daily_pnl: Dict[str, float] = {}

    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss_price: float,
    ) -> Dict:
        """
        Tính kích thước vị thế dựa trên rủi ro

        Args:
            capital: Vốn hiện tại
            entry_price: Giá vào lệnh
            stop_loss_price: Giá stop loss

        Returns:
            Dict với position size và thông tin
        """
        max_risk = capital * self.config["max_risk_per_trade"]
        risk_per_unit = abs(entry_price - stop_loss_price)

        if risk_per_unit == 0:
            risk_per_unit = entry_price * self.config["default_stop_loss"]

        # Position size theo risk
        position_size = max_risk / risk_per_unit
        position_value = position_size * entry_price

        # Giới hạn tối đa 30% vốn
        max_position_value = capital * 0.3
        if position_value > max_position_value:
            position_size = max_position_value / entry_price
            position_value = max_position_value

        return {
            "size": round(position_size, 6),
            "value": round(position_value, 2),
            "risk_amount": round(max_risk, 2),
            "risk_percent": round(self.config["max_risk_per_trade"] * 100, 2),
            "stop_loss": round(stop_loss_price, 2),
            "entry_price": round(entry_price, 2),
        }

    def calculate_stop_loss(self, entry_price: float, side: str,
                            atr: float = None) -> Dict:
        """
        Tính stop-loss và take-profit

        Args:
            entry_price: Giá vào lệnh
            side: "buy" hoặc "sell"
            atr: ATR value (nếu có, dùng ATR-based SL)

        Returns:
            Dict với SL/TP prices
        """
        if atr and atr > 0:
            # ATR-based stops (2x ATR for SL, 3x ATR for TP)
            sl_distance = atr * 2
            tp_distance = atr * 3
        else:
            sl_distance = entry_price * self.config["default_stop_loss"]
            tp_distance = entry_price * self.config["default_take_profit"]

        if side == "buy":
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        return {
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "sl_distance": round(sl_distance, 2),
            "tp_distance": round(tp_distance, 2),
            "risk_reward_ratio": round(tp_distance / sl_distance, 2) if sl_distance > 0 else 0,
        }

    def check_risk_limits(
        self,
        capital: float,
        current_positions: int,
        daily_pnl: float = 0,
        peak_equity: float = None,
    ) -> Dict:
        """
        Kiểm tra giới hạn rủi ro

        Returns:
            Dict với trạng thái rủi ro
        """
        can_trade = True
        warnings = []

        # Check max positions
        if current_positions >= self.config["max_open_positions"]:
            can_trade = False
            warnings.append(f"Đã đạt số lệnh tối đa ({self.config['max_open_positions']})")

        # Check daily loss limit
        daily_loss_limit = self.initial_capital * self.config["daily_loss_limit"]
        if daily_pnl < -daily_loss_limit:
            can_trade = False
            warnings.append(f"Vượt giới hạn lỗ trong ngày (${daily_loss_limit:.0f})")

        # Check max drawdown — tính từ peak equity (không phải initial capital)
        reference = peak_equity if peak_equity and peak_equity > 0 else self.initial_capital
        drawdown = (reference - capital) / reference
        if drawdown > self.config["max_drawdown"]:
            can_trade = False
            warnings.append(f"Vượt drawdown tối đa ({self.config['max_drawdown']*100:.0f}%)")

        # Remaining capital check
        if capital < self.initial_capital * 0.1:
            can_trade = False
            warnings.append("Vốn còn lại dưới 10%")

        return {
            "can_trade": can_trade,
            "warnings": warnings,
            "current_drawdown": round(drawdown * 100, 2),
            "daily_pnl": round(daily_pnl, 2),
            "open_positions": current_positions,
            "max_positions": self.config["max_open_positions"],
        }

    def calculate_portfolio_metrics(
        self,
        equity_curve: List[float],
        returns: List[float] = None,
    ) -> Dict:
        """
        Tính các chỉ số rủi ro portfolio

        Args:
            equity_curve: Danh sách giá trị vốn theo thời gian
            returns: Danh sách returns (nếu có)

        Returns:
            Dict chứa metrics
        """
        if not equity_curve or len(equity_curve) < 2:
            return {}

        equity = np.array(equity_curve)

        if returns is None:
            returns = np.diff(equity) / equity[:-1]
        else:
            returns = np.array(returns)

        # Total return
        total_return = (equity[-1] - equity[0]) / equity[0] * 100

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        max_drawdown = np.max(drawdown)

        # Sharpe Ratio (annualized, assuming hourly data)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24)
        else:
            sharpe = 0

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(365 * 24)
        else:
            sortino = 0

        # Win rate
        if len(returns) > 0:
            win_rate = np.sum(returns > 0) / len(returns) * 100
        else:
            win_rate = 0

        # Profit factor
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        return {
            "total_return": round(total_return, 2),
            "max_drawdown": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "total_trades": len(returns),
            "current_equity": round(equity[-1], 2),
        }
