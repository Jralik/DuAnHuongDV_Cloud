"""
Quick Start Script - Chạy nhanh hệ thống
"""
import uvicorn
import config

if __name__ == "__main__":
    print("\n>> Khoi dong General AI Trading System...\n")
    uvicorn.run(
        "main:app",
        host=config.SERVER_CONFIG["host"],
        port=config.SERVER_CONFIG["port"],
        reload=config.SERVER_CONFIG["reload"],
    )
