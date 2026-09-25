"""
AI/ML Weather Forecasting Module for AtmosBlend AI
Trains Random Forest Regressors for Rainfall, Temperature, and Wind Speed.
Performs Train/Test splitting, calculates MAE and RMSE, extracts feature importance,
and serializes models for real-time inference.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

class AIWeatherModel:
    def __init__(self, data_path: str = None, models_dir: str = None):
        self.data_path = data_path or os.path.join("data", "processed", "weather_cleaned.csv")
        self.models_dir = models_dir or "models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.models: Dict[str, RandomForestRegressor] = {}
        self.feature_columns = [
            "Latitude", "Longitude", "Lead_Time_Hours",
            "Month", "Hour", "Hour_Sin", "Hour_Cos", "Month_Sin", "Month_Cos"
        ]
        self.categorical_columns = ["Region", "Season", "Weather_Regime"]
        self.all_feature_cols = []
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.feature_importances: Dict[str, Dict[str, float]] = {}

    def prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        # One-hot encode categorical variables
        df_encoded = pd.get_dummies(df[self.feature_columns + self.categorical_columns], 
                                    columns=self.categorical_columns, 
                                    drop_first=False)
        
        if is_training:
            self.all_feature_cols = df_encoded.columns.tolist()
        else:
            # Reindex to ensure identical feature columns
            for col in self.all_feature_cols:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            df_encoded = df_encoded[self.all_feature_cols]

        targets = {}
        if "Actual_Rainfall" in df.columns:
            targets["Rainfall"] = df["Actual_Rainfall"]
        if "Actual_Temperature" in df.columns:
            targets["Temperature"] = df["Actual_Temperature"]
        if "Actual_Wind" in df.columns:
            targets["Wind"] = df["Actual_Wind"]

        return df_encoded, targets

    def train_models(self) -> Dict[str, Any]:
        df = pd.read_csv(self.data_path)
        X, targets = self.prepare_features(df, is_training=True)
        
        results = {}
        
        target_configs = {
            "Rainfall": {"n_estimators": 100, "max_depth": 14, "random_state": 42},
            "Temperature": {"n_estimators": 100, "max_depth": 12, "random_state": 42},
            "Wind": {"n_estimators": 100, "max_depth": 12, "random_state": 42}
        }
        
        for var_name, y in targets.items():
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42
            )
            
            cfg = target_configs[var_name]
            rf = RandomForestRegressor(**cfg)
            rf.fit(X_train, y_train)
            
            y_pred_train = rf.predict(X_train)
            y_pred_test = rf.predict(X_test)
            
            train_mae = float(mean_absolute_error(y_train, y_pred_train))
            train_rmse = float(root_mean_squared_error(y_train, y_pred_train))
            test_mae = float(mean_absolute_error(y_test, y_pred_test))
            test_rmse = float(root_mean_squared_error(y_test, y_pred_test))
            
            self.models[var_name] = rf
            self.metrics[var_name] = {
                "train_mae": round(train_mae, 2),
                "train_rmse": round(train_rmse, 2),
                "test_mae": round(test_mae, 2),
                "test_rmse": round(test_rmse, 2)
            }
            
            # Save top 10 feature importances
            importances = dict(zip(X.columns, rf.feature_importances_))
            sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True)[:10])
            self.feature_importances[var_name] = {k: round(float(v), 4) for k, v in sorted_importances.items()}
            
            # Save model to disk
            model_file = os.path.join(self.models_dir, f"rf_{var_name.lower()}.joblib")
            joblib.dump(rf, model_file)
            
        # Save feature metadata
        metadata = {
            "all_feature_cols": self.all_feature_cols,
            "feature_columns": self.feature_columns,
            "categorical_columns": self.categorical_columns,
            "metrics": self.metrics,
            "feature_importances": self.feature_importances
        }
        joblib.dump(metadata, os.path.join(self.models_dir, "model_metadata.joblib"))
        
        return {
            "metrics": self.metrics,
            "feature_importances": self.feature_importances,
            "models_saved": list(self.models.keys())
        }

    def load_saved_models(self) -> bool:
        meta_file = os.path.join(self.models_dir, "model_metadata.joblib")
        if not os.path.exists(meta_file):
            return False
            
        metadata = joblib.load(meta_file)
        self.all_feature_cols = metadata["all_feature_cols"]
        self.metrics = metadata["metrics"]
        self.feature_importances = metadata["feature_importances"]
        
        for var_name in ["Rainfall", "Temperature", "Wind"]:
            model_file = os.path.join(self.models_dir, f"rf_{var_name.lower()}.joblib")
            if os.path.exists(model_file):
                self.models[var_name] = joblib.load(model_file)
        return True

    def predict(self, single_row_dict: Dict[str, Any]) -> Dict[str, float]:
        if not self.models:
            loaded = self.load_saved_models()
            if not loaded:
                self.train_models()
                
        df_input = pd.DataFrame([single_row_dict])
        
        # Add cyclical features if not present
        if "Hour" in df_input.columns:
            df_input["Hour_Sin"] = np.sin(2 * np.pi * df_input["Hour"] / 24.0)
            df_input["Hour_Cos"] = np.cos(2 * np.pi * df_input["Hour"] / 24.0)
        else:
            df_input["Hour"] = 12
            df_input["Hour_Sin"] = 0.0
            df_input["Hour_Cos"] = -1.0
            
        if "Month" in df_input.columns:
            df_input["Month_Sin"] = np.sin(2 * np.pi * df_input["Month"] / 12.0)
            df_input["Month_Cos"] = np.cos(2 * np.pi * df_input["Month"] / 12.0)
        else:
            df_input["Month"] = 7
            df_input["Month_Sin"] = 0.0
            df_input["Month_Cos"] = -1.0
            
        if "Lead_Time_Hours" not in df_input.columns and "Lead_Time" in df_input.columns:
            lt_map = {"6 hours": 6, "12 hours": 12, "24 hours": 24, "48 hours": 48, "72 hours": 72}
            df_input["Lead_Time_Hours"] = df_input["Lead_Time"].map(lt_map).fillna(24)

        X, _ = self.prepare_features(df_input, is_training=False)
        
        predictions = {}
        for var_name, model in self.models.items():
            pred_val = float(model.predict(X)[0])
            if var_name == "Rainfall":
                pred_val = max(0.0, round(pred_val, 1))
            elif var_name == "Wind":
                pred_val = max(1.0, round(pred_val, 1))
            else:
                pred_val = round(pred_val, 1)
            predictions[var_name] = pred_val
            
        return predictions

def main():
    ai_engine = AIWeatherModel()
    print("Training AI/ML Weather Prediction Models (Random Forest)...")
    res = ai_engine.train_models()
    print("\n=== AI/ML MODEL PERFORMANCE (TEST SET) ===")
    for var, m in res["metrics"].items():
        print(f"Target: {var:<12} | Test MAE: {m['test_mae']:<6} | Test RMSE: {m['test_rmse']:<6}")
    print("\nModels successfully saved to 'models/' directory.")

if __name__ == "__main__":
    main()
