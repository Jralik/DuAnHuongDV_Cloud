"""
General AI Trading System - Main Application
FastAPI server + Trading Engine
"""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import os

import config
from core.data_fetcher import DataManager
from core.strategy import TradingStrategy
from core.backtester import Backtester
from core.portfolio import Portfolio
from api.routes import router, set_engine
from api.websocket import ws_manager, start_price_stream, start_signal_stream


class TradingEngine:
    """
    Trading Engine chính - điều phối tất cả components
    """

    def __init__(self):
        self.symbols = config.SYMBOLS
        self.data_manager = DataManager()
        self.strategy = TradingStrategy()
        self.portfolio = Portfolio(config.INITIAL_CAPITAL)
        self.is_running = False

    def get_analysis(self, symbol: str) -> dict:
        """Phân tích tổng hợp cho symbol"""
        df = self.data_manager.get_data(symbol, config.DEFAULT_TIMEFRAME, 200)
        return self.strategy.get_analysis(symbol, df)

    def train_model(self, symbol: str, days: int = 90) -> dict:
        """Train AI model"""
        print(f"[INFO] Đang train model cho {symbol} ({days} ngày)...")
        df = self.data_manager.fetcher.fetch_historical_data(
            symbol, config.DEFAULT_TIMEFRAME, days
        )
        if df.empty:
            return {"error": "Không lấy được dữ liệu lịch sử"}

        from datetime import datetime
        # Tạo tên model dựa trên dữ liệu và thời gian hiện tại
        model_name = f"{symbol.replace('/', '_')}_{days}days_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        result = self.strategy.train_model(symbol, df, model_name)
        result["model_name"] = model_name
        print(f"[INFO] Train model hoàn tất: {result}")
        return result

    def load_model(self, symbol: str, model_name: str) -> dict:
        """Load mô hình AI đã train từ thư mục"""
        print(f"[INFO] Đang load model {model_name} cho {symbol}...")
        predictor = self.strategy.get_predictor(symbol)
        try:
            predictor.load_models(symbol, model_name=model_name)
            if predictor.is_trained:
                return {"status": "success", "message": f"Đã tải {model_name}"}
            else:
                return {"error": "Load thất bại (model chưa trained hoặc lỗi file)"}
        except Exception as e:
            return {"error": f"Lỗi khi load model: {str(e)}"}

    def run_backtest(self, symbol: str, days: int = 90,
                     strategy: str = "ai") -> dict:
        """Chạy backtest"""
        print(f"[INFO] Đang chạy backtest {symbol} ({days} ngày, strategy={strategy})...")
        df = self.data_manager.fetcher.fetch_historical_data(
            symbol, config.DEFAULT_TIMEFRAME, days
        )
        if df.empty:
            return {"error": "Không lấy được dữ liệu lịch sử"}

        backtester = Backtester()
        result = backtester.run(df, symbol, config.INITIAL_CAPITAL, strategy)
        print(f"[INFO] Backtest hoàn tất: {result.get('total_return', 'N/A')}% return")
        return result


# Trading Engine instance
engine = TradingEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle"""
    print("=" * 60)
    print("  >> GENERAL AI TRADING SYSTEM")
    print(f"  >> Symbols: {', '.join(engine.symbols)}")
    print(f"  >> Von: ${engine.portfolio.initial_capital:,.0f}")
    print(f"  >> Dashboard: http://localhost:{config.SERVER_CONFIG['port']}")

    # Databricks initialization
    if config.DATABRICKS_CONFIG.get("enabled"):
        try:
            from db_integration.client import get_client
            from db_integration.data_pipeline import get_pipeline
            client = get_client()
            if client:
                status = client.test_connection()
                if status.get("connected"):
                    print(f"  >> Databricks: Connected ({config.DATABRICKS_CONFIG['host']})")
                    # Setup Delta Lake tables
                    pipeline = get_pipeline()
                    if pipeline:
                        setup = pipeline.setup_tables()
                        if setup.get("success"):
                            print("  >> Delta Lake: Tables ready")
                        else:
                            print(f"  >> Delta Lake: {setup.get('error', 'Setup failed')}")
                else:
                    print(f"  >> Databricks: Lỗi kết nối - {status.get('error', 'Unknown')}")
        except Exception as e:
            print(f"  >> Databricks: {e}")

        # MLflow initialization
        try:
            from db_integration.mlflow_tracker import get_tracker
            tracker = get_tracker()
            if tracker and tracker.is_ready:
                print(f"  >> MLflow: Ready ({config.MLFLOW_CONFIG['experiment_name']})")
            else:
                print("  >> MLflow: Không khởi tạo được")
        except Exception as e:
            print(f"  >> MLflow: {e}")
    else:
        print("  >> Databricks: Disabled (chưa cấu hình)")

    print("=" * 60)

    # Set engine for routes
    set_engine(engine)
    engine.is_running = True

    # Start background tasks
    price_task = asyncio.create_task(start_price_stream(engine))
    signal_task = asyncio.create_task(start_signal_stream(engine))

    yield

    # Shutdown
    engine.is_running = False
    price_task.cancel()
    signal_task.cancel()
    engine.portfolio.save_state()

    # Close Databricks connection
    if config.DATABRICKS_CONFIG.get("enabled"):
        try:
            from db_integration.client import get_client
            client = get_client()
            if client:
                client.close()
        except Exception:
            pass

    print("[INFO] Hệ thống đã dừng.")


# Create FastAPI app
app = FastAPI(
    title="General AI Trading System",
    description="Hệ thống giao dịch AI tổng hợp",
    version="1.0.0",
    lifespan=lifespan,
)

# Include API routes
app.include_router(router)

# Static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve dashboard"""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>General AI Trading</h1><p>Dashboard đang được tải...</p>")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint cho real-time updates"""
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle client messages if needed
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
