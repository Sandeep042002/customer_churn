import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb
from data_processing import Preprocessing
from utils import get_logger, save_model, load_model

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self, df: pd.DataFrame):
        if df is None or df.empty:
            logger.error("Invalid or empty DataFrame provided.")
            raise ValueError("Dataframe cannot be None or empty.")

        self.df = df
        self.input_data = df.drop(columns=["y"], errors="ignore")
        self.target = df.get("y")

        if self.target is None:
            logger.error("Target column 'y' is missing in the dataset.")
            raise ValueError("Target column 'y' is missing in the dataset.")

        self.rf_model = RandomForestClassifier(random_state=42)
        self.xgb_model = xgb.XGBClassifier(random_state=42)

    def split_data(self):
        logger.info("Splitting the data into train and test sets")
        return train_test_split(
            self.input_data,
            self.target,
            test_size=0.2,
            random_state=42,
            stratify=self.target,
        )

    def train_model(self, model, X_train, y_train, name):
        logger.info(f"Training {name} model...")
        model.fit(X_train, y_train)
        return model

    def tune_model(self, model, params, X_train, y_train, name):
        logger.info(f"Tuning {name} model...")
        grid_search = GridSearchCV(model, params, cv=3, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        logger.info(f"Best parameters for {name}: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def evaluate_model(self, model, X_test, y_test, name):
        logger.info(f"Evaluating {name} model...")
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        logger.info(f"{name} Accuracy: {accuracy:.4f}")
        logger.info(
            f"{name} Classification Report:\n{classification_report(y_test, preds)}"
        )

    def run(self):
        X_train, X_test, y_train, y_test = self.split_data()

        # Train base models
        rf_trained = self.train_model(self.rf_model, X_train, y_train, "Random Forest")
        xgb_trained = self.train_model(self.xgb_model, X_train, y_train, "XGBoost")

        # Hyperparameter tuning
        rf_params = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        }
        xgb_params = {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.1, 0.01],
        }

        rf_tuned = self.tune_model(
            rf_trained, rf_params, X_train, y_train, "Random Forest"
        )
        xgb_tuned = self.tune_model(
            xgb_trained, xgb_params, X_train, y_train, "XGBoost"
        )

        # Evaluation
        self.evaluate_model(rf_tuned, X_test, y_test, "Random Forest")
        self.evaluate_model(xgb_tuned, X_test, y_test, "XGBoost")

        # save_model(rf_tuned, "random_forest_model.pkl")
        save_model(xgb_tuned, "xgboost_model.pkl")
        logger.info("✅ Models saved successfully!")


if __name__ == "__main__":
    df = pd.read_csv(r"artifacts\files\md5\09\0e523ed03600022c5450ffe7463781")
    pre = Preprocessing()
    pro_df = pre.processing_data(df, is_training=True)  # Ensure training mode

    if pro_df is None:
        raise ValueError(
            "Error: Preprocessing function returned None. Check the implementation."
        )

    trainer = ModelTrainer(pro_df)
    trainer.run()
