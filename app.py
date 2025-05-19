from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load the model
model = joblib.load("model.pkl")

# Create the FastAPI app
app = FastAPI()

# Input data model
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Add all necessary features

@app.get("/")
def home():
    logging.info("Home endpoint called")
    return {"message": "Welcome to the ML model API"}

@app.get("/health")
def health_check():
    logging.info("Health check endpoint called")
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    logging.info(f"Predict endpoint called with data: {data}")
    features = np.array([[data.feature1, data.feature2, data.feature3]])  # Extend as needed
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
