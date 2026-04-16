"""
API Routes - FastAPI endpoints cho trading system
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from datetime import datetime

router = APIRouter()

# Trading engine instance sẽ được inject từ main.py
trading_engine = None


def set_engine(engine):
    """Set trading engine instance"""
    global trading_engine
    trading_engine = engine


@router.get("/api/status")
async def get_status():
    """Trạng thái hệ thống"""
    if not trading_engine:
        return {"status": "not_initialized"}
    return {
        "status": "running",
        "symbols": trading_engine.symbols,
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/api/market/{symbol_base}/{symbol_quote}")
async def get_market_data(
    symbol_base: str,
    symbol_quote: str,
    timeframe: str = Query("1h", description="Timeframe"),
    limit: int = Query(200, description="Số nến"),
):
    """Lấy dữ liệu thị trường"""
    if not trading_engine:
        raise HTTPException(500, "Engine chưa khởi tạo")

    symbol = f"{symbol_base}/{symbol_quote}"
    try:
        df = trading_engine.data_manager.get_data(symbol, timeframe, limit)
        if df.empty:
            raise HTTPException(404, f"Không có dữ liệu cho {symbol}")

        # Convert to JSON-friendly format
        data = []
        for idx, row in df.iterrows():
            data.append({
                "time": idx.isoformat(),
                "timestamp": int(idx.timestamp() * 1000),
                "open": round(row["open"], 2),
                "high": round(row["high"], 2),
                "low": round(row["low"], 2),
                "close": round(row["close"], 2),
                "volume": round(row["volume"], 4),
            })

        return {"symbol": symbol, "timeframe": timeframe, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/api/analysis/{symbol_base}/{symbol_quote}")
async def get_analysis(symbol_base: str, symbol_quote: str):
    """Phân tích kỹ thuật + AI cho symbol"""
    if not trading_engine:
        raise HTTPException(500, "Engine chưa khởi tạo")

    symbol = f"{symbol_base}/{symbol_quote}"
    try:
        analysis = trading_engine.get_analysis(symbol)
        return analysis
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/api/train/{symbol_base}/{symbol_quote}")
async def train_model(
    symbol_base: str,
    symbol_quote: str,
    days: int = Query(90, description="Số ngày dữ liệu"),
):
    """Train AI model cho symbol"""
    if not trading_engine:
        raise HTTPException(500, "Engine chưa khởi tạo")

    symbol = f"{symbol_base}/{symbol_quote}"
    try:
        result = trading_engine.train_model(symbol, days)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

@router.get("/api/models/{symbol_base}/{symbol_quote}")
async def list_models(symbol_base: str, symbol_quote: str):
    """Danh sách các model hiện có"""
    symbol = f"{symbol_base}/{symbol_quote}"
    try:
        from core.models import get_available_models
        models = get_available_models(symbol)
        return {"symbol": symbol, "models": models}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/api/models/load/{symbol_base}/{symbol_quote}")
async def load_model(
    symbol_base: str,
    symbol_quote: str,
    model_name: str = Query(..., description="Tên model (VD: BTC_USDT_180days_...)"),
):
    """Load model"""
    if not trading_engine:
        raise HTTPException(500, "Engine chưa khởi tạo")

    symbol = f"{symbol_base}/{symbol_quote}"
    try:
        result = trading_engine.load_model(symbol, model_name)
        if "error" in result:
             raise HTTPException(400, result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@router.delete("/api/models/{symbol_base}/{symbol_quote}")
async def delete_model(
    symbol_base: str,
    symbol_quote: str,
    model_name: str = Query(..., description="Tên model")
):
    """Xóa model đã lưu"""
    import os
    import shutil
    import config

    symbol = f"{symbol_base}/{symbol_quote}"
    symbol_clean = symbol.replace("/", "_")
    
    # Path traversal protection
    if not model_name or ".." in model_name or "/" in model_name or "\\" in model_name:
        raise HTTPException(400, "Tên model không hợp lệ")
        
    if not model_name.startswith(symbol_clean):
        raise HTTPException(400, "Không thể xóa model của symbol khác")

    base_path = os.path.join(config.MODELS_DIR, model_name)
    
    if not os.path.exists(base_path) or not os.path.isdir(base_path):
        raise HTTPException(404, "Không tìm thấy model")
        
    try:
        shutil.rmtree(base_path)
        return {"status": "success", "message": f"Đã xóa {model_name}"}
    except Exception as e:
        raise HTTPException(500, f"Lỗi khi xóa: {str(e)}")


@router.get("/api/signals")
async def get_signals(
    symbol: Optional[str] = None,
    limit: int = Query(20, description="Số tín hiệu"),
):
    """Lấy lịch sử tín hiệu"""
    if not trading_engine:
        raise HTTPException(500, "Engine chưa khởi tạo")

    return {
        "signals": trading_engine.strategy.get_signal_history(symbol, limit)
    }


@router.get("/api/portfolio")
async def get_portfolio():
    """Tổng quan portfolio"""
    if not trading_engine:
        raise HTTPException(500, "Engine chưa khởi tạo")

    return trading_engine.portfolio.get_summary()


@router.post("/api/trade/open")
async def open_trade(
    symbol: str,
    side: str = Query("buy", description="buy hoặc sell"),
):
    """Mở vị thế (paper trading)"""
    if not trading_engine:
        raise HTTPException(500, "Engine chưa khởi tạo")

    try:
        ticker = trading_engine.data_manager.fetcher.fetch_ticker(symbol)
        price = ticker["last"]
        if price <= 0:
            raise HTTPException(400, "Không lấy được giá")

        # Get ATR for SL/TP
        df = trading_engine.data_manager.get_data(symbol, "1h", 100)
        atr = None
        if not df.empty:
            from core.indicators import TechnicalIndicators
            ti = TechnicalIndicators()
            df_ind = ti.calculate_all(df)
            if "atr" in df_ind.columns:
                atr = float(df_ind["atr"].iloc[-1])

        result = trading_engine.portfolio.open_position(symbol, side, price, atr=atr)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/api/trade/close/{symbol_base}/{symbol_quote}")
async def close_trade(symbol_base: str, symbol_quote: str):
    """Đóng vị thế"""
    if not trading_engine:
        raise HTTPException(500, "Engine chưa khởi tạo")

    symbol = f"{symbol_base}/{symbol_quote}"
    try:
        ticker = trading_engine.data_manager.fetcher.fetch_ticker(symbol)
        price = ticker["last"]
        result = trading_engine.portfolio.close_position(symbol, price, "Thủ công")
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/api/backtest/{symbol_base}/{symbol_quote}")
async def run_backtest(
    symbol_base: str,
    symbol_quote: str,
    days: int = Query(90, description="Số ngày"),
    strategy: str = Query("ai", description="ai hoặc technical"),
):
    """Chạy backtest"""
    if not trading_engine:
        raise HTTPException(500, "Engine chưa khởi tạo")

    symbol = f"{symbol_base}/{symbol_quote}"
    try:
        result = trading_engine.run_backtest(symbol, days, strategy)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/api/prices")
async def get_prices():
    """Lấy giá mới nhất cho tất cả symbols"""
    if not trading_engine:
        raise HTTPException(500, "Engine chưa khởi tạo")

    try:
        prices = trading_engine.data_manager.get_latest_prices()
        return {"prices": prices}
    except Exception as e:
        raise HTTPException(500, str(e))


# ============================================================
# DATABRICKS ENDPOINTS
# ============================================================

@router.get("/api/databricks/status")
async def databricks_status():
    """Kiểm tra trạng thái kết nối Databricks"""
    import config
    if not config.DATABRICKS_CONFIG.get("enabled"):
        return {
            "enabled": False,
            "message": "Databricks chưa được cấu hình. Thêm DATABRICKS_HOST vào .env",
        }

    try:
        from db_integration.client import get_client
        from db_integration.mlflow_tracker import get_tracker
        from db_integration.data_pipeline import get_pipeline

        result = {"enabled": True}

        # Test SQL connection
        client = get_client()
        if client:
            conn_status = client.test_connection()
            result["sql_warehouse"] = conn_status
        else:
            result["sql_warehouse"] = {"connected": False}

        # MLflow status
        tracker = get_tracker()
        if tracker:
            result["mlflow"] = tracker.get_status()
        else:
            result["mlflow"] = {"enabled": False}

        # Data pipeline sync status
        pipeline = get_pipeline()
        if pipeline:
            result["data_sync"] = pipeline.get_sync_status()
        else:
            result["data_sync"] = {"enabled": False}

        return result

    except Exception as e:
        return {"enabled": True, "error": str(e)}


@router.post("/api/databricks/sync-data")
async def sync_data_to_databricks(
    symbol: str = Query("BTC/USDT", description="Symbol"),
    timeframe: str = Query("1h", description="Timeframe"),
    days: int = Query(30, description="Số ngày dữ liệu"),
):
    """Đồng bộ dữ liệu OHLCV lên Databricks Delta Lake"""
    import config
    if not config.DATABRICKS_CONFIG.get("enabled"):
        raise HTTPException(400, "Databricks chưa được cấu hình")

    if not trading_engine:
        raise HTTPException(500, "Engine chưa khởi tạo")

    try:
        from db_integration.data_pipeline import get_pipeline

        # Lấy dữ liệu từ exchange
        df = trading_engine.data_manager.fetcher.fetch_historical_data(
            symbol, timeframe, days
        )
        if df.empty:
            raise HTTPException(404, f"Không lấy được dữ liệu cho {symbol}")

        # Sync lên Databricks
        pipeline = get_pipeline()
        if not pipeline:
            raise HTTPException(500, "Không thể khởi tạo data pipeline")

        result = pipeline.sync_ohlcv(symbol, df, timeframe)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/api/databricks/sync-trades")
async def sync_trades_to_databricks():
    """Đồng bộ lịch sử giao dịch lên Databricks Delta Lake"""
    import config
    if not config.DATABRICKS_CONFIG.get("enabled"):
        raise HTTPException(400, "Databricks chưa được cấu hình")

    if not trading_engine:
        raise HTTPException(500, "Engine chưa khởi tạo")

    try:
        from db_integration.data_pipeline import get_pipeline

        trades = [t.to_dict() for t in trading_engine.portfolio.trade_history]
        pipeline = get_pipeline()
        if not pipeline:
            raise HTTPException(500, "Không thể khởi tạo data pipeline")

        result = pipeline.sync_trades(trades)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/api/databricks/sync-signals")
async def sync_signals_to_databricks():
    """Đồng bộ lịch sử tín hiệu AI lên Databricks Delta Lake"""
    import config
    if not config.DATABRICKS_CONFIG.get("enabled"):
        raise HTTPException(400, "Databricks chưa được cấu hình")

    if not trading_engine:
        raise HTTPException(500, "Engine chưa khởi tạo")

    try:
        from db_integration.data_pipeline import get_pipeline

        # Lấy tối đa 100 tín hiệu gần nhất
        signals = [s.to_dict() for s in trading_engine.strategy.signal_history]
        
        pipeline = get_pipeline()
        if not pipeline:
            raise HTTPException(500, "Không thể khởi tạo data pipeline")

        result = pipeline.sync_signals(signals)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/api/databricks/experiments")
async def get_mlflow_experiments(
    limit: int = Query(20, description="Số runs tối đa"),
):
    """Lấy danh sách MLflow experiment runs"""
    import config
    if not config.MLFLOW_CONFIG.get("enabled"):
        return {"enabled": False, "runs": []}

    try:
        from db_integration.mlflow_tracker import get_tracker
        tracker = get_tracker()
        if not tracker:
            return {"enabled": False, "runs": []}

        runs = tracker.get_experiment_history(limit)
        return {
            "enabled": True,
            "experiment_name": tracker.experiment_name,
            "runs": runs,
            "total": len(runs),
        }

    except Exception as e:
        return {"enabled": True, "error": str(e), "runs": []}


@router.get("/api/databricks/best-model")
async def get_best_model(
    metric: str = Query("ensemble_avg_accuracy", description="Metric để so sánh"),
):
    """Lấy model tốt nhất từ MLflow"""
    import config
    if not config.MLFLOW_CONFIG.get("enabled"):
        return {"enabled": False}

    try:
        from db_integration.mlflow_tracker import get_tracker
        tracker = get_tracker()
        if not tracker:
            return {"enabled": False}

        best = tracker.get_best_run(metric)
        return {
            "enabled": True,
            "metric": metric,
            "best_run": best,
        }

    except Exception as e:
        return {"enabled": True, "error": str(e)}


@router.get("/api/databricks/training-history")
async def get_training_history(
    symbol: str = Query(None, description="Lọc theo symbol"),
    limit: int = Query(20, description="Số kết quả tối đa"),
):
    """Lấy lịch sử training từ Delta Lake"""
    import config
    if not config.DATABRICKS_CONFIG.get("enabled"):
        return {"enabled": False, "history": []}

    try:
        from db_integration.data_pipeline import get_pipeline
        pipeline = get_pipeline()
        if not pipeline:
            return {"enabled": False, "history": []}

        history = pipeline.get_training_history(symbol, limit)
        return {
            "enabled": True,
            "history": history,
            "total": len(history),
        }

    except Exception as e:
        return {"enabled": True, "error": str(e), "history": []}
