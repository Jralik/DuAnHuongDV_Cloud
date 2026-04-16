"""
MLflow Tracker - Experiment tracking trên Databricks
Log training runs, metrics, parameters, và model artifacts
"""
import os
from typing import Dict, List, Optional
from datetime import datetime

import config


class MLflowTracker:
    """
    MLflow experiment tracking trên Databricks.
    Log: parameters, metrics, artifacts cho mỗi training run.
    """

    def __init__(self):
        self.tracking_uri = config.MLFLOW_CONFIG.get("tracking_uri", "databricks")
        self.experiment_name = config.MLFLOW_CONFIG.get(
            "experiment_name", "/Shared/AI_Trading_Experiments"
        )
        self._initialized = False
        self._mlflow = None
        self._init_mlflow()

    def _init_mlflow(self):
        """Khởi tạo MLflow với Databricks tracking URI"""
        try:
            import mlflow

            # Set environment variables cho Databricks auth
            os.environ["DATABRICKS_HOST"] = f"https://{config.DATABRICKS_CONFIG['host']}"
            os.environ["DATABRICKS_TOKEN"] = config.DATABRICKS_CONFIG["token"]

            mlflow.set_tracking_uri(self.tracking_uri)

            # Tạo hoặc lấy experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    mlflow.create_experiment(self.experiment_name)
            except Exception as e:
                print(f"[MLFLOW] Cảnh báo khi tạo experiment: {e}")

            mlflow.set_experiment(self.experiment_name)
            self._mlflow = mlflow
            self._initialized = True
            print(f"[MLFLOW] Đã khởi tạo — Experiment: {self.experiment_name}")

        except ImportError:
            print("[MLFLOW] mlflow chưa được cài đặt")
            self._initialized = False
        except Exception as e:
            print(f"[MLFLOW] Lỗi khởi tạo: {e}")
            self._initialized = False

    @property
    def is_ready(self) -> bool:
        return self._initialized and self._mlflow is not None

    def start_training_run(self, symbol: str, days: int,
                           params: Dict = None) -> Optional[str]:
        """
        Bắt đầu một MLflow training run.

        Args:
            symbol: Cặp giao dịch
            days: Số ngày dữ liệu
            params: Training parameters

        Returns:
            run_id hoặc None nếu lỗi
        """
        if not self.is_ready:
            return None

        try:
            with self._mlflow.start_run(
                run_name=f"{symbol}_{days}d_{datetime.now().strftime('%m%d_%H%M')}",
                tags={
                    "symbol": symbol,
                    "days": str(days),
                    "training_date": datetime.now().isoformat(),
                    "system": "AI_Trading",
                },
            ) as run:
                run_id = run.info.run_id

                # Log basic parameters
                self._mlflow.log_param("symbol", symbol)
                self._mlflow.log_param("days", days)
                self._mlflow.log_param("timeframe", config.DEFAULT_TIMEFRAME)

                # Log model config params
                if params:
                    model_config = params.get("model_config", {})

                    # XGBoost params
                    xgb_config = model_config.get("xgboost", {})
                    for key in ["n_estimators", "max_depth", "learning_rate", "subsample"]:
                        if key in xgb_config:
                            self._mlflow.log_param(f"xgb_{key}", xgb_config[key])

                    # Random Forest params
                    rf_config = model_config.get("random_forest", {})
                    for key in ["n_estimators", "max_depth", "min_samples_split"]:
                        if key in rf_config:
                            self._mlflow.log_param(f"rf_{key}", rf_config[key])

                    # LSTM params
                    lstm_config = model_config.get("lstm", {})
                    for key in ["hidden_size", "num_layers", "dropout", "epochs", "learning_rate"]:
                        if key in lstm_config:
                            self._mlflow.log_param(f"lstm_{key}", lstm_config[key])

                    # Ensemble weights
                    weights = model_config.get("ensemble_weights", {})
                    for model_name, weight in weights.items():
                        self._mlflow.log_param(f"weight_{model_name}", weight)

                    # Feature count
                    if "features_count" in params:
                        self._mlflow.log_param("features_count", params["features_count"])

            print(f"[MLFLOW] Started run: {run_id}")
            return run_id

        except Exception as e:
            print(f"[MLFLOW] Lỗi start_run: {e}")
            return None

    def log_model_metrics(self, run_id: str, model_name: str,
                          metrics: Dict) -> bool:
        """
        Log metrics cho một model cụ thể.

        Args:
            run_id: MLflow run ID
            model_name: Tên model (xgboost, random_forest, lstm)
            metrics: Dict chứa accuracy, f1, etc.
        """
        if not self.is_ready:
            return False

        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient(tracking_uri=self.tracking_uri)
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    client.log_metric(run_id, f"{model_name}_{key}", value)
            return True

        except Exception as e:
            print(f"[MLFLOW] Lỗi log_metrics ({model_name}): {e}")
            return False

    def log_ensemble_metrics(self, run_id: str, metrics: Dict) -> bool:
        """Log metrics tổng hợp cho ensemble"""
        if not self.is_ready:
            return False

        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient(tracking_uri=self.tracking_uri)
            for model_name, model_metrics in metrics.items():
                if isinstance(model_metrics, dict) and "error" not in model_metrics:
                    for key, value in model_metrics.items():
                        if isinstance(value, (int, float)):
                            client.log_metric(run_id, f"{model_name}_{key}", value)

            # Tính ensemble accuracy trung bình
            accuracies = []
            for model_metrics in metrics.values():
                if isinstance(model_metrics, dict) and "accuracy" in model_metrics:
                    accuracies.append(model_metrics["accuracy"])
            if accuracies:
                client.log_metric(run_id, "ensemble_avg_accuracy",
                                        round(sum(accuracies) / len(accuracies), 4))
            return True

        except Exception as e:
            print(f"[MLFLOW] Lỗi log_ensemble_metrics: {e}")
            return False

    def log_model_artifact(self, run_id: str, model_dir: str) -> bool:
        """Upload model files lên MLflow artifacts"""
        if not self.is_ready:
            return False

        try:
            if os.path.exists(model_dir) and os.path.isdir(model_dir):
                from mlflow.tracking import MlflowClient
                client = MlflowClient(tracking_uri=self.tracking_uri)
                client.log_artifacts(run_id, model_dir, "model_files")
                print(f"[MLFLOW] Đã upload artifacts từ {model_dir}")
                return True
            return False

        except Exception as e:
            print(f"[MLFLOW] Lỗi log_artifact: {e}")
            return False

    def end_run(self, run_id: str, status: str = "FINISHED") -> bool:
        """Kết thúc một MLflow run"""
        if not self.is_ready:
            return False

        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient(tracking_uri=self.tracking_uri)
            client.log_param(run_id, "status", status)
            client.log_param(run_id, "end_time", datetime.now().isoformat())
            client.set_terminated(run_id, status=status)
            print(f"[MLFLOW] Run {run_id[:8]}... ended: {status}")
            return True

        except Exception as e:
            print(f"[MLFLOW] Lỗi end_run: {e}")
            return False

    def get_experiment_history(self, max_results: int = 20) -> List[Dict]:
        """Lấy danh sách runs của experiment hiện tại"""
        if not self.is_ready:
            return []

        try:
            experiment = self._mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return []

            from mlflow.entities import ViewType
            runs = self._mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_results,
                order_by=["start_time DESC"],
            )

            if runs.empty:
                return []

            results = []
            for _, run in runs.iterrows():
                run_dict = {
                    "run_id": run.get("run_id", ""),
                    "run_name": run.get("tags.mlflow.runName", ""),
                    "status": run.get("status", ""),
                    "start_time": str(run.get("start_time", "")),
                    "symbol": run.get("params.symbol", ""),
                    "days": run.get("params.days", ""),
                }

                # Extract metrics
                for col in runs.columns:
                    if col.startswith("metrics."):
                        metric_name = col.replace("metrics.", "")
                        val = run.get(col)
                        if pd.notna(val):
                            run_dict[metric_name] = round(float(val), 4)

                results.append(run_dict)

            return results

        except Exception as e:
            print(f"[MLFLOW] Lỗi get_experiment_history: {e}")
            return []

    def get_best_run(self, metric: str = "ensemble_avg_accuracy") -> Optional[Dict]:
        """Lấy run có metric tốt nhất"""
        if not self.is_ready:
            return None

        try:
            experiment = self._mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return None

            runs = self._mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=1,
                order_by=[f"metrics.{metric} DESC"],
            )

            if runs.empty:
                return None

            run = runs.iloc[0]
            result = {
                "run_id": run.get("run_id", ""),
                "run_name": run.get("tags.mlflow.runName", ""),
                "symbol": run.get("params.symbol", ""),
                "days": run.get("params.days", ""),
            }

            for col in runs.columns:
                if col.startswith("metrics."):
                    metric_name = col.replace("metrics.", "")
                    val = run.get(col)
                    if pd.notna(val):
                        result[metric_name] = round(float(val), 4)

            return result

        except Exception as e:
            print(f"[MLFLOW] Lỗi get_best_run: {e}")
            return None

    def get_status(self) -> Dict:
        """Trạng thái MLflow tracker"""
        return {
            "enabled": config.MLFLOW_CONFIG.get("enabled", False),
            "initialized": self._initialized,
            "tracking_uri": self.tracking_uri,
            "experiment_name": self.experiment_name,
        }


# Singleton instance
_tracker_instance: Optional[MLflowTracker] = None


def get_tracker() -> Optional[MLflowTracker]:
    """Lấy singleton MLflowTracker instance"""
    global _tracker_instance
    if not config.MLFLOW_CONFIG.get("enabled"):
        return None
    if _tracker_instance is None:
        _tracker_instance = MLflowTracker()
    return _tracker_instance


# Cần import pandas cho get_experiment_history
try:
    import pandas as pd
except ImportError:
    pass
