"""
Trading Strategy - Chiến lược giao dịch
Kết hợp AI predictions với Technical Analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

import config
from core.indicators import TechnicalIndicators
from core.features import FeatureEngineer
from core.models import EnsemblePredictor


class Signal:
    """Đại diện một tín hiệu giao dịch"""

    def __init__(self, symbol: str, action: str, confidence: float,
                 price: float, reason: str, timestamp: datetime = None):
        self.symbol = symbol
        self.action = action  # "MUA", "BÁN", "GIỮ"
        self.confidence = confidence
        self.price = price
        self.reason = reason
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "confidence": round(self.confidence, 2),
            "price": self.price,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


class TradingStrategy:
    """
    Chiến lược giao dịch kết hợp AI + Technical Analysis

    Pipeline:
    1. Lấy dữ liệu → 2. Tính indicators → 3. Feature engineering →
    4. AI prediction → 5. Signal generation → 6. Risk check
    """

    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.feature_engineer = FeatureEngineer()
        self.predictors: Dict[str, EnsemblePredictor] = {}
        self.signal_history: List[Signal] = []

    def get_predictor(self, symbol: str) -> EnsemblePredictor:
        """Lấy hoặc tạo predictor cho symbol"""
        if symbol not in self.predictors:
            self.predictors[symbol] = EnsemblePredictor()
        return self.predictors[symbol]

    def train_model(self, symbol: str, df: pd.DataFrame, model_name: str = None) -> Dict:
        """
        Train AI model cho một symbol

        Args:
            symbol: Cặp giao dịch
            df: DataFrame OHLCV thô
            model_name: Tên thư mục lưu model

        Returns:
            Training metrics
        """
        if len(df) < config.MODEL_CONFIG["min_data_points"]:
            return {"error": f"Không đủ dữ liệu. Cần tối thiểu {config.MODEL_CONFIG['min_data_points']} nến, hiện có {len(df)}"}

        # === MLflow: Start tracking run ===
        tracker = None
        run_id = None
        if config.MLFLOW_CONFIG.get("enabled"):
            try:
                from db_integration.mlflow_tracker import get_tracker
                tracker = get_tracker()
                if tracker:
                    run_id = tracker.start_training_run(symbol, len(df), {
                        "model_config": config.MODEL_CONFIG,
                        "features_count": 0,  # Updated later
                    })
            except Exception as e:
                print(f"[MLFLOW] Lỗi khởi tạo run: {e}")

        # 1. Tính indicators
        df_indicators = self.indicators.calculate_all(df)

        # 2. Feature engineering
        df_features = self.feature_engineer.create_features(df_indicators)

        # 3. Chuẩn bị training data
        X, y = self.feature_engineer.prepare_training_data(df_features)

        if X.empty:
            if tracker and run_id:
                tracker.end_run(run_id, "FAILED")
            return {"error": "Không thể tạo features cho training"}

        # 4. Scale features
        X_scaled = self.feature_engineer.scale_features(X, fit=True)

        # 5. Train ensemble (truyền run_id cho MLflow logging)
        predictor = self.get_predictor(symbol)
        predictor.feature_engineer = self.feature_engineer
        predictor._current_run_id = run_id  # Pass MLflow run ID
        metrics = predictor.train(X_scaled, y)

        # 6. Save models
        predictor.save_models(symbol, model_name)

        result = {
            "symbol": symbol,
            "status": "success",
            "data_points": len(X),
            "features": len(X.columns),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        # === MLflow: Log artifacts & end run ===
        if tracker and run_id:
            try:
                import os
                model_dir = os.path.join(config.MODELS_DIR, model_name or "")
                if os.path.exists(model_dir):
                    tracker.log_model_artifact(run_id, model_dir)
                tracker.end_run(run_id, "FINISHED")
            except Exception as e:
                print(f"[MLFLOW] Lỗi kết thúc run: {e}")

        # === Databricks: Sync training result ===
        if config.DATABRICKS_CONFIG.get("enabled"):
            try:
                from db_integration.data_pipeline import get_pipeline
                pipeline = get_pipeline()
                if pipeline:
                    result_with_name = {**result, "model_name": model_name or "", "days": len(df) // 24}
                    pipeline.sync_training_result(result_with_name)
            except Exception as e:
                print(f"[DATABRICKS] Lỗi sync training result: {e}")

        return result

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Signal:
        """
        Tạo tín hiệu giao dịch

        Args:
            symbol: Cặp giao dịch
            df: DataFrame OHLCV mới nhất

        Returns:
            Signal object
        """
        if df.empty:
            return Signal(symbol, "GIỮ", 0, 0, "Không có dữ liệu")

        # 1. Tính indicators
        df_indicators = self.indicators.calculate_all(df)

        # 2. Lấy tín hiệu technical
        tech_signals = self.indicators.get_latest_signals(df_indicators)

        # 3. Feature engineering
        df_features = self.feature_engineer.create_features(df_indicators)

        # 4. AI Prediction
        predictor = self.get_predictor(symbol)
        ai_prediction = {"signal": "GIỮ", "confidence": 0}

        if predictor.is_trained and self.feature_engineer._is_fitted:
            try:
                # Get features for prediction
                X_latest = self.feature_engineer.get_latest_features(df_features)

                if not X_latest.empty:
                    # Full scaled data for LSTM
                    exclude_cols = ["open", "high", "low", "close", "volume", "target"]
                    feature_cols = [c for c in df_features.columns
                                   if c not in exclude_cols
                                   and c in self.feature_engineer.feature_columns]
                    X_full = df_features[feature_cols].fillna(0)
                    if self.feature_engineer._is_fitted:
                        for col in self.feature_engineer.feature_columns:
                            if col not in X_full.columns:
                                X_full[col] = 0
                        X_full = X_full[self.feature_engineer.feature_columns]
                        X_full_scaled = pd.DataFrame(
                            self.feature_engineer.scaler.transform(X_full),
                            columns=X_full.columns,
                            index=X_full.index
                        )
                    else:
                        X_full_scaled = X_full

                    ai_prediction = predictor.predict(X_latest, X_full_scaled)
            except Exception as e:
                ai_prediction = {"signal": "GIỮ", "confidence": 0, "error": str(e)}

        # 5. Kết hợp tín hiệu
        signal = self._combine_signals(symbol, tech_signals, ai_prediction, df)

        # Check if signal action changed
        signal_changed = False
        if not self.signal_history:
            signal_changed = True
        else:
            last_signal = self.signal_history[-1]
            if last_signal.action != signal.action:
                signal_changed = True

        self.signal_history.append(signal)
        # Giữ tối đa 100 signals
        if len(self.signal_history) > 100:
            self.signal_history = self.signal_history[-100:]

        # Auto-sync signal to Databricks using a background thread (to avoid blocking) if it changed
        if signal_changed and config.DATABRICKS_CONFIG.get("enabled"):
            try:
                from db_integration.data_pipeline import get_pipeline
                pipeline = get_pipeline()
                if pipeline and signal.action != "GIỮ": # Option: Có thể bỏ qua "GIỮ" nếu chỉ quan tâm lệnh mua/bán
                    import threading
                    threading.Thread(
                        target=pipeline.sync_signals,
                        args=([signal.to_dict()],),
                        daemon=True
                    ).start()
            except Exception as e:
                print(f"[DATABRICKS] Lỗi auto-sync signal: {e}")

        return signal

    def _combine_signals(self, symbol: str, tech_signals: Dict,
                         ai_prediction: Dict, df: pd.DataFrame) -> Signal:
        """
        Kết hợp tín hiệu technical và AI.
        - Tăng confidence khi AI và Technical đồng thuận (consensus bonus)
        - High Confidence Zone khi tất cả chỉ báo cùng chiều
        """
        price = float(df.iloc[-1]["close"])
        reasons = []

        # Score tổng hợp (-100 đến +100)
        total_score = 0
        weight_sum = 0

        # ── AI signal (weight: 55%) ──────────────────────────────
        ai_signal = ai_prediction.get("signal", "GIỮ")
        ai_conf = ai_prediction.get("confidence", 0)
        ai_prob = ai_prediction.get("probability", 0.5)

        if ai_signal == "MUA":
            total_score += ai_conf * 0.55
            reasons.append(f"AI: MUA ({ai_conf:.1f}%)")
        elif ai_signal == "BÁN":
            total_score -= ai_conf * 0.55
            reasons.append(f"AI: BÁN ({ai_conf:.1f}%)")
        else:
            reasons.append(f"AI: GIỮ ({ai_conf:.1f}%)")
        weight_sum += 0.55

        # ── Technical momentum score (weight: 25%) ───────────────
        momentum = tech_signals.get("momentum_score", 0)
        # Clamp momentum về [-100, 100]
        momentum = max(-100, min(100, momentum))
        total_score += momentum * 0.25
        weight_sum += 0.25

        # ── RSI signal (weight: 10%) ─────────────────────────────
        rsi = tech_signals.get("rsi", 50)
        if rsi < 30:
            total_score += 40 * 0.10
            reasons.append(f"RSI: Quá bán ({rsi:.0f})")
        elif rsi > 70:
            total_score -= 40 * 0.10
            reasons.append(f"RSI: Quá mua ({rsi:.0f})")
        weight_sum += 0.10

        # ── MACD signal (weight: 10%) ────────────────────────────
        macd_hist = tech_signals.get("macd_histogram", 0)
        if macd_hist > 0:
            total_score += 25 * 0.10
            reasons.append("MACD: Tăng")
        else:
            total_score -= 25 * 0.10
            reasons.append("MACD: Giảm")
        weight_sum += 0.10

        # Normalize
        normalized_score = total_score / weight_sum if weight_sum > 0 else 0
        confidence = abs(normalized_score)

        # ── CONSENSUS BONUS ──────────────────────────────────────
        # Tăng confidence khi AI và Momentum đồng thuận cùng chiều
        ai_bullish = ai_signal == "MUA"
        ai_bearish = ai_signal == "BÁN"
        tech_bullish = momentum > 20
        tech_bearish = momentum < -20
        macd_bullish = macd_hist > 0
        macd_bearish = macd_hist < 0

        all_bullish = ai_bullish and tech_bullish and macd_bullish and rsi < 65
        all_bearish = ai_bearish and tech_bearish and macd_bearish and rsi > 35

        if all_bullish or all_bearish:
            # High Confidence Zone: tất cả đồng thuận → boost +25%
            confidence = min(confidence * 1.35, 100)
            reasons.append("HIGH_CONF: Đồng thuận")
        elif (ai_bullish and tech_bullish) or (ai_bearish and tech_bearish):
            # Partial consensus: AI + Momentum cùng chiều → boost nhẹ
            confidence = min(confidence * 1.15, 100)

        # ── Determine action (ngưỡng giảm xuống 12 để nhạy hơn) ──
        if normalized_score > 12 and confidence > 15:
            action = "MUA"
        elif normalized_score < -12 and confidence > 15:
            action = "BÁN"
        else:
            action = "GIỮ"

        reason_str = " | ".join(reasons)
        return Signal(symbol, action, confidence, price, reason_str)


    def get_signal_history(self, symbol: str = None, limit: int = 20) -> List[Dict]:
        """Lấy lịch sử tín hiệu"""
        signals = self.signal_history
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
        return [s.to_dict() for s in signals[-limit:]]

    def get_analysis(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Phân tích tổng hợp cho symbol"""
        if df.empty:
            return {"error": "Không có dữ liệu"}

        df_indicators = self.indicators.calculate_all(df)
        tech_signals = self.indicators.get_latest_signals(df_indicators)
        signal = self.generate_signal(symbol, df)

        predictor = self.get_predictor(symbol)

        return {
            "symbol": symbol,
            "price": float(df.iloc[-1]["close"]),
            "change_24h": float(df["close"].pct_change(24).iloc[-1] * 100) if len(df) > 24 else 0,
            "signal": signal.to_dict(),
            "technical": tech_signals,
            "ai_trained": predictor.is_trained,
            "ai_info": predictor.training_info,
            "timestamp": datetime.now().isoformat(),
        }
