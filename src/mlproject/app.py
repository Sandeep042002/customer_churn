from fastapi import FastAPI
import pandas as pd
from components.constant import PredictionRequest
from components.prediction import PredictionPipeline

app = FastAPI(
    title="Bank Marketing Predictor",
    description="API for predicting bank marketing campaign success",
    version="1.0.0",
)

# Define model paths
ENCODER_PATH = "artifacts/encoder.pkl"
MODEL_PATH = "artifacts/xgboost_model.pkl"
SCALER_PATH = "artifacts/scaler.pkl"
BOXCOX = "artifacts/boxcox_lambda.pkl"


preprocessor = PredictionPipeline(
    model_path=MODEL_PATH,
    encoder_path=ENCODER_PATH,
    scaler_path=SCALER_PATH,
    boxcox_lambda_path=BOXCOX,
)


@app.get("/")
async def root():
    return {"message": "Welcome to the Bank Marketing Predictor API"}


@app.post("/predict/")
def predict(data: PredictionRequest):
    try:
        input_df = pd.DataFrame([data.model_dump()])
        prediction = preprocessor.predict(input_df)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": f"❌ Prediction failed: {str(e)}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
