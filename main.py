from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import logging
import uvicorn
import numpy as np

# Logging setup
logging.basicConfig(filename="logs/app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load model and transformer
model = joblib.load("/content/MLOps-Course/main.py")
transformer = joblib.load("/content/MLOps-Course/column_transformer.pkl")

# App init
app = FastAPI()

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

import pandas as pd

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
        transformed = transformer.transform(input_df)
        prediction = model.predict(transformed)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)