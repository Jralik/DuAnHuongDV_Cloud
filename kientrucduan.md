# SƠ ĐỒ KIẾN TRÚC TỔNG THỂ DỰ ÁN 
**Tên hệ thống:** MẠNG LƯỚI GIAO DỊCH AI TỔNG HỢP TRÊN NỀN TẢNG HYBRID-CLOUD
**Mô hình áp dụng:** Kiến trúc hướng dịch vụ (SOA/Microservices)

---

## 1. MÔ HÌNH KIẾN TRÚC HƯỚNG DỊCH VỤ (SOA)
Hệ thống được thiết kế theo dạng **Lỏng lẻo (Loosely Coupled) kiểu Hybrid-Cloud**, bao gồm các luồng Dịch vụ (Services) giao tiếp chuyên biệt:
* **Frontend Service (Client-side):** Trạm tương tác đồ họa (UI) với người dùng. Nhận luồng dữ liệu thời gian thực mà không thực hiện tính toán nặng.
* **API Gateway Service (FastAPI):** Lõi phân phối dịch vụ trung tâm. Cung cấp chuẩn giao tiếp chuẩn RESTful API và giao thức WebSocket hai chiều.
* **Trading & AI Engine Service (Local Edge Worker):** Đóng vai trò là cỗ máy cày cuốc, liên tục thực hiện nghiệp vụ cào dữ liệu bên ngoài (Binance) rút trích đặc trưng, và chạy mô hình học máy theo vòng lặp milli-giây.
* **Storage & MLOps Service (Databricks Cloud):** Dịch vụ máy chủ đám mây, đóng vai trò như một Kho Dữ Liệu khổng lồ (Data Lake) và Trung tâm Giám sát Phiên Trí Tuệ Nhân Tạo (MLflow Server).

---

## 2. SƠ ĐỒ KIẾN TRÚC TỔNG THỂ (MERMAID)

```mermaid
flowchart TD
    %% Định nghĩa các Style cho đẹp
    classDef frontend fill:#3b82f6,stroke:#1d4ed8,stroke-width:2px,color:#fff
    classDef backend fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff
    classDef cloud fill:#f59e0b,stroke:#b45309,stroke-width:2px,color:#fff
    classDef external fill:#6b7280,stroke:#374151,stroke-width:2px,color:#fff

    %% 1. Frontend Service
    subgraph Client ["💻 1. Giao diện (Frontend Service)"]
        UI["Dashboard Web UI\n(JS/HTML/CSS)"]
    end
    class Client frontend

    %% 2. API Gateway & Core Engine
    subgraph Local_App ["🖥️ 2. App Local (Trading & Gateway Service)"]
        API["FastAPI Gateway\nRouter & Endpoints"]
        WS["WebSocket Stream\n(Bi-directional)"]
        
        subgraph Engine ["Trading Core Engine"]
            DF["Data Fetcher\n(CCXT)"]
            TA["Technical Analysis\n(Indicators)"]
            ML["Machine Learning Env\n(Ensemble: XGBoost, LSTM)"]
            PF["Portfolio & Paper Trading"]
            Thread["Background Sync Thread\n(Auto-sync)"]
        end
        
        API <--> WS
        WS <--> Engine
    end
    class Local_App backend
    class UI,API,WS,DF,TA,ML,PF,Thread frontend

    %% 3. External API
    subgraph Ext ["🌐 3. Public Exchange"]
        Binance["Binance / Sàn Giao dịch\n(Market OHLCV)"]
    end
    class Ext external

    %% 4. Databricks Cloud
    subgraph Cloud ["☁️ 4. Nền tảng Đám Mây (Databricks Services)"]
        SQL["Databricks SQL Warehouse\n(Delta Lake)"]
        MLflow["Databricks MLflow\n(Experiment Tracking)"]
        
        T1[("ohlcv_data")]
        T2[("ai_signals")]
        T3[("trade_history")]
        T4[("training_history")]
        
        SQL --- T1 & T2 & T3 & T4
    end
    class Cloud cloud

    %% Hướng luồng kết nối Data (Data Flow)
    UI <==>|1. REST APIs (HTTP)| API
    UI <==>|2. Real-time Price/Signals| WS
    DF <==>|3. Fetch OHLCV| Binance
    
    %% Engine Internal Flow
    DF --> TA --> ML --> PF
    ML -. "Update Model" .-> PF
    TA & ML & PF --> Thread
    
    %% Cloud Intergration Flow
    Thread == "4. SQL Query (INSERT INTO)" ==> SQL
    ML == "5. Log Params / Metrics / Artifacts" ==> MLflow
```

---

## 3. GIẢI THÍCH CHI TIẾT LUỒNG DỮ LIỆU VÀ TÍCH HỢP (DATA FLOW & INTEGRATION)
1. **Flow Lấy dữ liệu (Ingestion):** Data Fetcher (Trong hệ thống Local) gọi API bên ngoài từ Sàn Binance để giật thông số nến (OHLCV) về với nhịp độ liên tục.
2. **Flow Xử lý (Calculation):** Tín hiệu được truyền qua một chuỗi lắp ráp (Pipeline): Tính biểu đồ (Technical Analysis) 👉 Làm nét tính năng (Feature Engineering) 👉 Đưa cho 3 thuật toán AI (ML Env) tính toán 👉 Trả về quyết định đánh lệnh (Portfolio).
3. **Flow Trình chiếu (Streaming):** Mọi thay đổi về chỉ số và ví tiền lúc này sẽ được truyền một cách tức thời nhờ khe cắm **WebSocket** hai chiều đẩy thẳng lên màn hình người dùng trên App. 
4. **Flow Đám mây (Cloud Integration):** Thiết kế tự động hóa đẩy dữ liệu lên nền tảng đám mây lớn (Big Data). 
    - Cứ mỗi khi AI ra lệnh mới, hoặc người chơi Chốt vị thế, tiến trình **Background Sync Thread** lập tức đứng ra đảm nhận việc tổng hợp thông tin, gọi HTTP Request để đóng gói nén thẳng lên Databricks Delta Lake. 
    - Mỗi khi ấn nút Train Model, hệ thống gọi API của `Databricks MLflow` để thu thập bảng điểm Accuracy và lưu trữ tệp não bộ thuật toán (`.pkl`) lên nền tảng để bảo quản và tạo nhật ký mô hình (Artifacts & Logging).
