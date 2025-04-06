import sys
import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from scipy.stats import boxcox
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from utils import get_logger, save_model, load_model

logger = get_logger(__name__)


class Preprocessing:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.encoder = None
        self.boxcox_lambda = None

    def processing_data(self, df: pd.DataFrame, is_training=True):
        logger.info("🔄 Starting data processing...")

        # Rename columns to match training data
        df.rename(columns={"conda age": "conda_age"}, inplace=True)

        df["dur"] = df["dur"].replace(0, 0.01)
        df["num_calls"] = df["num_calls"].replace(0, 0.01)

        if df["dur"].nunique() > 1:
            if is_training:
                transformed_dur, self.boxcox_lambda = boxcox(df["dur"] + 1)
                df["dur"] = transformed_dur
                save_model(self.boxcox_lambda, "boxcox_lambda.pkl")
                logger.info("saving the transforming data")

            else:
                self.boxcox_lambda = load_model("boxcox_lambda.pkl")
                df["dur"] = boxcox(df["dur"] + 1, lmbda=self.boxcox_lambda)

        df["conda_age"] = np.sqrt(df["conda_age"])
        df["num_calls"] = np.log1p(df["num_calls"])

        # One-Hot Encoding
        categorical_cols = [
            "job",
            "marital",
            "education_qual",
            "call_type",
            "mon",
            "prev_outcome",
        ]

        if is_training:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoded_array = self.encoder.fit_transform(df[categorical_cols])
            save_model(self.encoder, "encoder.pkl")
            logger.info
        else:
            self.encoder = load_model("encoder.pkl")
            encoded_array = self.encoder.transform(df[categorical_cols])

        encoded_df = pd.DataFrame(
            encoded_array,
            columns=self.encoder.get_feature_names_out(categorical_cols),
            index=df.index,
        )
        df = df.drop(columns=categorical_cols)
        df = pd.concat([df, encoded_df], axis=1)

        # Scaling
        numeric_cols = ["conda_age", "dur", "num_calls"]

        if is_training:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            save_model(self.scaler, "scaler.pkl")
        else:
            self.scaler = load_model("scaler.pkl")
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        # Label encoding for target (optional, only if 'y' is present)

        if "y" in df.columns and df["y"].dtype == object:
            df["y"] = df["y"].str.lower().map({"yes": 1, "no": 0})

        logger.info("✅ Data processing completed successfully!")
        return df


# Training Mode
if __name__ == "__main__":
    df = pd.read_csv(r"D:\customer_predictio\Data\customer_call.csv")
    preprocessor = Preprocessing()
    processed_df = preprocessor.processing_data(df, is_training=True)

    if processed_df is not None:
        print(processed_df.head())
        print(processed_df.columns)
    else:
        print("⚠️ No valid data to process.")
