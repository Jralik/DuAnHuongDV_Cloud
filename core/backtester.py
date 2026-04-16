"""
Backtester - Backtesting chiến lược trên dữ liệu lịch sử
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

import config
from core.indicators import TechnicalIndicators
from core.features import FeatureEngineer
from core.models import EnsemblePredictor
from core.risk_manager import RiskManager


class Trade:
    """Đại diện một giao dịch"""

    def __init__(self, symbol: str, side: str, entry_price: float,
                 size: float, entry_time: datetime):
        self.symbol = symbol
        self.side = side  # "buy" hoặc "sell"
        self.entry_price = entry_price
        self.size = size
        self.entry_time = entry_time
        self.exit_price: Optional[float] = None
        self.exit_time: Optional[datetime] = None
        self.pnl: float = 0
        self.status: str = "open"
        self.exit_reason: str = ""

    def close(self, exit_price: float, exit_time: datetime, reason: str = ""):
        """Đóng vị thế"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        self.status = "closed"

        if self.side == "buy":
            self.pnl = (exit_price - self.entry_price) * self.size
        else:
            self.pnl = (self.entry_price - exit_price) * self.size

        # Trừ phí
        commission = (self.entry_price + exit_price) * self.size * config.BACKTEST_CONFIG["commission"]
        self.pnl -= commission

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(self.exit_price, 2) if self.exit_price else None,
            "size": round(self.size, 6),
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl / (self.entry_price * self.size) * 100, 2) if self.entry_price * self.size > 0 else 0,
            "entry_time": self.entry_time.isoformat() if isinstance(self.entry_time, datetime) else str(self.entry_time),
            "exit_time": self.exit_time.isoformat() if isinstance(self.exit_time, datetime) else str(self.exit_time) if self.exit_time else None,
            "status": self.status,
            "exit_reason": self.exit_reason,
        }


class Backtester:
    """
    Backtesting engine
    Walk-forward testing với tính phí và slippage
    """

    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.feature_engineer = FeatureEngineer()
        self.risk_manager = RiskManager()
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.results: Dict = {}

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        initial_capital: float = None,
        strategy: str = "ai",  # "ai" hoặc "technical"
    ) -> Dict:
        """
        Chạy backtest

        Args:
            df: DataFrame OHLCV lịch sử
            symbol: Symbol
            initial_capital: Vốn ban đầu
            strategy: Loại chiến lược

        Returns:
            Dict kết quả backtest
        """
        capital = initial_capital or config.INITIAL_CAPITAL
        self.trades = []
        self.equity_curve = [capital]

        if len(df) < config.MODEL_CONFIG["min_data_points"]:
            return {"error": f"Không đủ dữ liệu. Cần {config.MODEL_CONFIG['min_data_points']}, có {len(df)}"}

        # 1. Tính indicators
        df_ind = self.indicators.calculate_all(df)
        df_feat = self.feature_engineer.create_features(df_ind)

        # 2. Chuẩn bị data
        X, y = self.feature_engineer.prepare_training_data(df_feat)
        if X.empty:
            return {"error": "Không thể tạo features"}

        X_scaled = self.feature_engineer.scale_features(X, fit=True)

        # 3. Walk-forward split
        train_size = int(len(X_scaled) * 0.6)
        test_start = train_size

        # 4. Train models on first 60%
        if strategy == "ai":
            predictor = EnsemblePredictor()
            X_train = X_scaled.iloc[:train_size]
            y_train = y.iloc[:train_size]
            predictor.train(X_train, y_train)

        # 5. Simulate trading on remaining 40%
        current_capital = capital
        open_trade: Optional[Trade] = None

        test_indices = list(range(test_start, len(X_scaled)))

        for i in test_indices:
            timestamp = X_scaled.index[i]
            # Get corresponding OHLCV data
            if timestamp not in df.index:
                continue

            current_price = float(df.loc[timestamp, "close"])
            current_high = float(df.loc[timestamp, "high"])
            current_low = float(df.loc[timestamp, "low"])

            # Check stop-loss / take-profit for open position
            if open_trade:
                sl_info = self.risk_manager.calculate_stop_loss(
                    open_trade.entry_price, open_trade.side
                )

                # Check SL
                if open_trade.side == "buy" and current_low <= sl_info["stop_loss"]:
                    open_trade.close(sl_info["stop_loss"], timestamp, "Stop Loss")
                    current_capital += open_trade.pnl
                    self.trades.append(open_trade)
                    open_trade = None
                elif open_trade.side == "sell" and current_high >= sl_info["stop_loss"]:
                    open_trade.close(sl_info["stop_loss"], timestamp, "Stop Loss")
                    current_capital += open_trade.pnl
                    self.trades.append(open_trade)
                    open_trade = None

                # Check TP
                if open_trade:
                    if open_trade.side == "buy" and current_high >= sl_info["take_profit"]:
                        open_trade.close(sl_info["take_profit"], timestamp, "Take Profit")
                        current_capital += open_trade.pnl
                        self.trades.append(open_trade)
                        open_trade = None
                    elif open_trade.side == "sell" and current_low <= sl_info["take_profit"]:
                        open_trade.close(sl_info["take_profit"], timestamp, "Take Profit")
                        current_capital += open_trade.pnl
                        self.trades.append(open_trade)
                        open_trade = None

            # Generate signal
            if strategy == "ai" and predictor.is_trained:
                row = X_scaled.iloc[[i]]
                prediction = predictor.predict(row)
                signal = prediction.get("signal", "GIỮ")
                confidence = prediction.get("confidence", 0)
            else:
                # Technical strategy
                row_data = df_feat.loc[timestamp] if timestamp in df_feat.index else None
                if row_data is not None:
                    momentum = row_data.get("momentum_score", 0) if hasattr(row_data, 'get') else 0
                    rsi = row_data.get("rsi", 50) if hasattr(row_data, 'get') else 50
                    if momentum > 20 and rsi < 70:
                        signal = "MUA"
                        confidence = abs(momentum)
                    elif momentum < -20 and rsi > 30:
                        signal = "BÁN"
                        confidence = abs(momentum)
                    else:
                        signal = "GIỮ"
                        confidence = 0
                else:
                    signal = "GIỮ"
                    confidence = 0

            # Execute signal
            if signal == "MUA" and confidence > 20 and open_trade is None:
                # Check risk
                risk_check = self.risk_manager.check_risk_limits(
                    current_capital, 0
                )
                if risk_check["can_trade"]:
                    sl_price = current_price * (1 - config.RISK_CONFIG["default_stop_loss"])
                    pos_info = self.risk_manager.calculate_position_size(
                        current_capital, current_price, sl_price
                    )
                    # Apply slippage
                    entry = current_price * (1 + config.BACKTEST_CONFIG["slippage"])
                    open_trade = Trade(symbol, "buy", entry, pos_info["size"], timestamp)

            elif signal == "BÁN" and open_trade and open_trade.side == "buy":
                exit_price = current_price * (1 - config.BACKTEST_CONFIG["slippage"])
                open_trade.close(exit_price, timestamp, "Signal BÁN")
                current_capital += open_trade.pnl
                self.trades.append(open_trade)
                open_trade = None

            # Update equity curve
            if open_trade:
                unrealized = (current_price - open_trade.entry_price) * open_trade.size
                self.equity_curve.append(current_capital + unrealized)
            else:
                self.equity_curve.append(current_capital)

        # Close any remaining position
        if open_trade:
            last_price = float(df.iloc[-1]["close"])
            open_trade.close(last_price, df.index[-1], "Kết thúc backtest")
            current_capital += open_trade.pnl
            self.trades.append(open_trade)

        # 6. Calculate results
        self.results = self._calculate_results(capital, current_capital)
        return self.results

    def _calculate_results(self, initial_capital: float,
                           final_capital: float) -> Dict:
        """Tính kết quả backtest"""
        closed_trades = [t for t in self.trades if t.status == "closed"]
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]

        total_trades = len(closed_trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0

        # Returns
        returns = [t.pnl / (t.entry_price * t.size) for t in closed_trades if t.entry_price * t.size > 0]

        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Portfolio metrics
        portfolio_metrics = self.risk_manager.calculate_portfolio_metrics(
            self.equity_curve
        )

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([abs(t.pnl) for t in losing_trades]) if losing_trades else 0

        return {
            "initial_capital": round(initial_capital, 2),
            "final_capital": round(final_capital, 2),
            "total_return": round((final_capital - initial_capital) / initial_capital * 100, 2),
            "total_pnl": round(final_capital - initial_capital, 2),
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "portfolio_metrics": portfolio_metrics,
            "equity_curve": self.equity_curve,
            "trades": [t.to_dict() for t in closed_trades[-50:]],
        }
