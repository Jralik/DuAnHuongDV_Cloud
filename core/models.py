"""
AI/ML Models - XGBoost, Random Forest, LSTM
Ensemble prediction cho trading signals
"""
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Tuple, Optional
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config


class XGBoostModel:
    """XGBoost classifier cho dự đoán xu hướng"""

    def __init__(self, params: Dict = None):
        self.params = params or config.MODEL_CONFIG["xgboost"]
        self.model = None
        self.is_trained = False
        self.metrics: Dict = {}

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """Train model"""
        params = {k: v for k, v in self.params.items()
                  if k not in ("random_state", "early_stopping_rounds")}
        early_stopping = self.params.get("early_stopping_rounds", 30)

        self.model = xgb.XGBClassifier(
            **params,
            random_state=self.params.get("random_state", 42),
        )

        if X_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.set_params(early_stopping_rounds=early_stopping)
        else:
            eval_set = [(X_train, y_train)]

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        self.is_trained = True

        # Calculate metrics
        if X_val is not None:
            y_pred = self.model.predict(X_val)
            self.metrics = {
                "accuracy": round(accuracy_score(y_val, y_pred), 4),
                "f1": round(f1_score(y_val, y_pred, zero_division=0), 4),
            }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability"""
        if not self.is_trained:
            return np.array([0.5])
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> Dict[str, float]:
        """Lấy feature importance"""
        if not self.is_trained:
            return {}
        importance = self.model.feature_importances_
        features = self.model.get_booster().feature_names
        if features is None:
            return {}
        return dict(sorted(
            zip(features, importance),
            key=lambda x: x[1], reverse=True
        )[:15])

    def save(self, path: str):
        if self.model:
            joblib.dump(self.model, path)

    def load(self, path: str):
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.is_trained = True


class RandomForestModel:
    """Random Forest classifier cho dự đoán xu hướng"""

    def __init__(self, params: Dict = None):
        self.params = params or config.MODEL_CONFIG["random_forest"]
        self.model = None
        self.is_trained = False
        self.metrics: Dict = {}

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """Train model"""
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X_train, y_train)
        self.is_trained = True

        if X_val is not None:
            y_pred = self.model.predict(X_val)
            self.metrics = {
                "accuracy": round(accuracy_score(y_val, y_pred), 4),
                "f1": round(f1_score(y_val, y_pred, zero_division=0), 4),
            }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            return np.array([0.5])
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        if self.model:
            joblib.dump(self.model, path)

    def load(self, path: str):
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.is_trained = True


class LSTMNet(nn.Module):
    """LSTM Neural Network architecture"""

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out.squeeze()


class LSTMModel:
    """LSTM model wrapper cho trading prediction"""

    def __init__(self, params: Dict = None):
        self.params = params or config.MODEL_CONFIG["lstm"]
        self.model = None
        self.is_trained = False
        self.metrics: Dict = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_sequences(self, X: np.ndarray, y: np.ndarray = None
                          ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Tạo sequences cho LSTM"""
        seq_len = self.params["sequence_length"]
        X_seq, y_seq = [], []

        for i in range(seq_len, len(X)):
            X_seq.append(X[i - seq_len:i])
            if y is not None:
                y_seq.append(y[i])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        return X_seq, y_seq

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """Train LSTM model"""
        X_np = X_train.values.astype(np.float32)
        y_np = y_train.values.astype(np.float32)

        X_seq, y_seq = self._create_sequences(X_np, y_np)
        if len(X_seq) < 10:
            return

        input_size = X_seq.shape[2]
        self.model = LSTMNet(
            input_size=input_size,
            hidden_size=self.params["hidden_size"],
            num_layers=self.params["num_layers"],
            dropout=self.params["dropout"],
        ).to(self.device)

        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.params["batch_size"], shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(),
                                      lr=self.params["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5
        )
        criterion = nn.BCELoss()
        patience = self.params.get("patience", 15)
        best_loss = float("inf")
        no_improve = 0

        self.model.train()
        for epoch in range(self.params["epochs"]):
            total_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            scheduler.step(avg_loss)

            # Early stopping
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        self.is_trained = True

        # Validation metrics
        if X_val is not None and y_val is not None:
            X_val_np = X_val.values.astype(np.float32)
            y_val_np = y_val.values.astype(np.float32)
            X_val_seq, y_val_seq = self._create_sequences(X_val_np, y_val_np)
            if len(X_val_seq) > 0:
                self.model.eval()
                with torch.no_grad():
                    X_val_t = torch.FloatTensor(X_val_seq).to(self.device)
                    preds = self.model(X_val_t).cpu().numpy()
                    y_pred = (preds > 0.5).astype(int)
                    self.metrics = {
                        "accuracy": round(accuracy_score(y_val_seq, y_pred), 4),
                        "f1": round(f1_score(y_val_seq, y_pred, zero_division=0), 4),
                    }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability"""
        if not self.is_trained or self.model is None:
            return np.array([0.5])

        X_np = X.values.astype(np.float32)
        seq_len = self.params["sequence_length"]

        if len(X_np) < seq_len:
            return np.array([0.5])

        X_seq = X_np[-seq_len:].reshape(1, seq_len, -1)
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(X_tensor).cpu().numpy()
        return pred

    def save(self, path: str):
        if self.model:
            torch.save(self.model.state_dict(), path)

    def load(self, path: str, input_size: int):
        if os.path.exists(path):
            self.model = LSTMNet(
                input_size=input_size,
                hidden_size=self.params["hidden_size"],
                num_layers=self.params["num_layers"],
                dropout=self.params["dropout"],
            ).to(self.device)
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.is_trained = True


class EnsemblePredictor:
    """
    Kết hợp 3 models: XGBoost + Random Forest + LSTM
    Weighted voting cho tín hiệu cuối cùng
    """

    def __init__(self):
        self.xgb_model = XGBoostModel()
        self.rf_model = RandomForestModel()
        self.lstm_model = LSTMModel()
        self.weights = config.MODEL_CONFIG["ensemble_weights"]
        self.is_trained = False
        self.training_info: Dict = {}
        self.feature_engineer = None  # Set externally
        self._current_run_id: Optional[str] = None  # MLflow run ID

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train tất cả models

        Returns:
            Dict chứa training metrics
        """
        test_size = config.MODEL_CONFIG["test_size"]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        results = {}

        # Train XGBoost
        try:
            self.xgb_model.train(X_train, y_train, X_val, y_val)
            results["xgboost"] = self.xgb_model.metrics
        except Exception as e:
            results["xgboost"] = {"error": str(e)}

        # Train Random Forest
        try:
            self.rf_model.train(X_train, y_train, X_val, y_val)
            results["random_forest"] = self.rf_model.metrics
        except Exception as e:
            results["random_forest"] = {"error": str(e)}

        # Train LSTM
        try:
            self.lstm_model.train(X_train, y_train, X_val, y_val)
            results["lstm"] = self.lstm_model.metrics
        except Exception as e:
            results["lstm"] = {"error": str(e)}

        self.is_trained = True
        self.training_info = {
            "timestamp": datetime.now().isoformat(),
            "data_points": len(X),
            "features_count": len(X.columns),
            "models": results,
        }

        # Log metrics lên MLflow nếu enabled
        if self._current_run_id and config.MLFLOW_CONFIG.get("enabled"):
            try:
                from db_integration.mlflow_tracker import get_tracker
                tracker = get_tracker()
                if tracker:
                    tracker.log_ensemble_metrics(self._current_run_id, results)
            except Exception as e:
                print(f"[MLFLOW] Lỗi log metrics: {e}")

        return results

    def predict(self, X: pd.DataFrame, X_full: pd.DataFrame = None) -> Dict:
        """
        Ensemble prediction

        Args:
            X: Features cho prediction (1 row, scaled)
            X_full: Full feature history cho LSTM

        Returns:
            Dict với prediction và confidence
        """
        predictions = {}
        weighted_prob = 0
        total_weight = 0

        # XGBoost prediction
        if self.xgb_model.is_trained:
            try:
                xgb_pred = self.xgb_model.predict(X)
                xgb_prob = float(xgb_pred[0]) if len(xgb_pred) > 0 else 0.5
                predictions["xgboost"] = round(xgb_prob, 4)
                weighted_prob += xgb_prob * self.weights["xgboost"]
                total_weight += self.weights["xgboost"]
            except Exception:
                predictions["xgboost"] = 0.5

        # Random Forest prediction
        if self.rf_model.is_trained:
            try:
                rf_pred = self.rf_model.predict(X)
                rf_prob = float(rf_pred[0]) if len(rf_pred) > 0 else 0.5
                predictions["random_forest"] = round(rf_prob, 4)
                weighted_prob += rf_prob * self.weights["random_forest"]
                total_weight += self.weights["random_forest"]
            except Exception:
                predictions["random_forest"] = 0.5

        # LSTM prediction
        if self.lstm_model.is_trained and X_full is not None:
            try:
                lstm_pred = self.lstm_model.predict(X_full)
                lstm_prob = float(lstm_pred[0]) if len(lstm_pred) > 0 else 0.5
                predictions["lstm"] = round(lstm_prob, 4)
                weighted_prob += lstm_prob * self.weights["lstm"]
                total_weight += self.weights["lstm"]
            except Exception:
                predictions["lstm"] = 0.5

        # Ensemble result
        if total_weight > 0:
            ensemble_prob = weighted_prob / total_weight
        else:
            ensemble_prob = 0.5

        # Determine signal
        threshold = config.MODEL_CONFIG["confidence_threshold"]
        if ensemble_prob > threshold:
            signal = "MUA"
        elif ensemble_prob < (1 - threshold):
            signal = "BÁN"
        else:
            signal = "GIỮ"

        # FIX: Rescale confidence đúng cách
        # Xác suất thực tế của ensemble nằm trong khoảng [0.40, 0.70]
        # Map khoảng [0.5, 0.75+] → [0, 100%] để hiển thị đúng
        deviation = abs(ensemble_prob - 0.5)
        # Rescale: 0.0 → 0%, 0.10 → 60%, 0.15+ → 90%+
        confidence = min(deviation * 600, 100.0)  # 600 = 100/0.1667

        return {
            "signal": signal,
            "probability": round(ensemble_prob, 4),
            "confidence": round(confidence, 2),
            "individual_predictions": predictions,
            "model_metrics": self.training_info.get("models", {}),
        }

    def save_models(self, symbol: str, model_name: str = None):
        """Lưu tất cả models và metadata"""
        symbol_clean = symbol.replace("/", "_")
        if not model_name:
            model_name = f"{symbol_clean}_default"
            
        base_path = os.path.join(config.MODELS_DIR, model_name)
        os.makedirs(base_path, exist_ok=True)

        self.xgb_model.save(os.path.join(base_path, "xgb.pkl"))
        self.rf_model.save(os.path.join(base_path, "rf.pkl"))
        self.lstm_model.save(os.path.join(base_path, "lstm.pth"))
        
        # Save metadata (input_size, info)
        import json
        with open(os.path.join(base_path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.training_info, f, ensure_ascii=False, indent=2)

    def load_models(self, symbol: str, model_name: str = None, input_size: int = 0):
        """Load models đã lưu"""
        symbol_clean = symbol.replace("/", "_")
        if not model_name:
            model_name = f"{symbol_clean}_default"
            
        base_path = os.path.join(config.MODELS_DIR, model_name)
        
        # Read metadata for input_size if not provided
        if input_size == 0 and os.path.exists(os.path.join(base_path, "meta.json")):
            import json
            try:
                with open(os.path.join(base_path, "meta.json"), "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    input_size = meta.get("features_count", 0)
                    self.training_info = meta
            except Exception as e:
                print(f"[ERROR] Loading meta.json: {e}")

        self.xgb_model.load(os.path.join(base_path, "xgb.pkl"))
        self.rf_model.load(os.path.join(base_path, "rf.pkl"))
        if input_size > 0:
            self.lstm_model.load(os.path.join(base_path, "lstm.pth"), input_size)

        self.is_trained = (
            self.xgb_model.is_trained or
            self.rf_model.is_trained or
            self.lstm_model.is_trained
        )

def get_available_models(symbol: str) -> list:
    """Lấy danh sách các models đã lưu cho một symbol"""
    symbol_clean = symbol.replace("/", "_")
    base_path = config.MODELS_DIR
    models = []
    if os.path.exists(base_path):
        for item in os.listdir(base_path):
            if os.path.isdir(os.path.join(base_path, item)) and item.startswith(symbol_clean):
                 models.append(item)
    # Sort by descending order (newest first based on naming convention)
    return sorted(models, reverse=True)
