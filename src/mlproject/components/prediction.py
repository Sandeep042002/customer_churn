import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from scipy.stats import boxcox
from components.utils import get_logger, load_model

logger = get_logger(__name__)


class PredictionPipeline:
    def __init__(self, model_path, encoder_path, scaler_path, boxcox_lambda_path):
        logger.info("Loading the models")
        self.model = load_model(model_path)
        self.encoder = load_model(encoder_path)
        self.scaler = load_model(scaler_path)
        self.boxcox_lambda = load_model(boxcox_lambda_path)

    def preprocess_data(self, df: pd.DataFrame):
        logger.info("Preprocessing new data...")

        # Ensure column names match
        df.rename(columns={"conda age": "conda_age"}, inplace=True)

        # Handle zeros
        df["dur"] = df["dur"].replace(0, 0.01)
        df["num_calls"] = df["num_calls"].replace(0, 0.01)

        # Transformations (apply same ones from training)
        if df["dur"].nunique() > 1:
            df["dur"] = boxcox(df["dur"] + 1, lmbda=self.boxcox_lambda)
        df["conda_age"] = np.sqrt(df["conda_age"])
        df["num_calls"] = np.log1p(df["num_calls"])

        # One-Hot Encoding
        logger.info("Encoding categorical variables...")
        categorical_cols = [
            "job",
            "marital",
            "education_qual",
            "call_type",
            "mon",
            "prev_outcome",
        ]
        encoded_array = self.encoder.transform(df[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=self.encoder.get_feature_names_out(categorical_cols),
            index=df.index,
        )
        df = df.drop(columns=categorical_cols)
        df = pd.concat([df, encoded_df], axis=1)

        # Scaling
        logger.info("Scaling numerical data...")
        numeric_cols = ["conda_age", "dur", "num_calls"]
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        expected_features = getattr(
            self.model, "feature_names_in_", df.columns.tolist()
        )
        df = df.reindex(columns=expected_features, fill_value=0)

        return df

    def predict(self, df: pd.DataFrame):
        processed_df = self.preprocess_data(df)
        predictions = self.model.predict(processed_df)
        return predictions


# ⬇️ Example usage
if __name__ == "__main__":
    df = pd.DataFrame(
        [
            {
                "conda_age": 20,
                "job": "student",
                "marital": "divorced",
                "education_qual": "secondary",
                "call_type": "cellular",
                "day": 1,
                "mon": "jul",
                "dur": 20,
                "num_calls": 1,
                "prev_outcome": "unknown",
            }
        ]
    )

    # Paths to saved models
    # encoder_path = r"D:\customer_predictio\artifacts/encoder.pkl"
    # model_path = r"D:\customer_predictio\artifacts/xgboost_model.pkl"
    # scaler_path = r"D:\customer_predictio\artifacts/scaler.pkl"
    # boxcox_lambda_path = r"D:\customer_predictio\artifacts\boxcox_lambda.pkl"
    encoder_path = "artifacts/encoder.pkl"
    model_path = "artifacts/xgboost_model.pkl"
    scaler_path = "artifacts/scaler.pkl"
    boxcox_lambda_path = "artifacts/boxcox_lambda.pkl"

    predictor = PredictionPipeline(
        model_path=model_path,
        encoder_path=encoder_path,
        scaler_path=scaler_path,
        boxcox_lambda_path=boxcox_lambda_path,
    )

    predictions = predictor.predict(df)
    print("Prediction:", predictions)
