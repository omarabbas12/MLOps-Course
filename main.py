from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import logging
import uvicorn
import numpy as np
import pandas as pd
import os
from prometheus_fastapi_instrumentator import Instrumentator

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Logging setup
logging.basicConfig(filename="logs/app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load model and transformer (using relative path from script directory)
model = joblib.load("model.pkl")
transformer = joblib.load("column_transformer.pkl")

# App init
app = FastAPI()

Instrumentator().instrument(app).expose(app)
# Define input schema
class ModelInput(BaseModel):
    CreditScore: float
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float
    Geography: str
    Gender: str


@app.get("/")
def home():
    logging.info("Home endpoint hit")
    return {"message": "Welcome to the GradBoost model API"}

@app.get("/health")
def health():
    logging.info("Health check endpoint hit")
    return {"status": "OK"}

@app.post("/predict")
def predict(data: ModelInput):
    try:
        input_df = pd.DataFrame([{
            "CreditScore": data.CreditScore,
            "Age": data.Age,
            "Tenure": data.Tenure,
            "Balance": data.Balance,
            "NumOfProducts": data.NumOfProducts,
            "HasCrCard": data.HasCrCard,
            "IsActiveMember": data.IsActiveMember,
            "EstimatedSalary": data.EstimatedSalary,
            "Geography": data.Geography,
            "Gender": data.Gender
        }])

        print("✅ Incoming DataFrame:")
        print(input_df)

        transformed = transformer.transform(input_df)
        prediction = model.predict(transformed)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ONLY run this locally, not in Colab
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
