from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Base correct input
sample_input = {
    "CreditScore": 600,
    "Age": 40,
    "Tenure": 5,
    "Balance": 50000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 100000,
    "Geography": "Germany",
    "Gender": "Male"
}


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the GradBoost model API"}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

def test_predict_valid_input():
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert "prediction" in response.json()
def test_predict_all_zero_inputs():
    zero_input = {key: 0 for key in sample_input}
    zero_input["Geography"] = "France"  # or any valid geography from training
    zero_input["Gender"] = "Female"     # must be a valid category
    response = client.post("/predict", json=zero_input)
    assert response.status_code == 200
    assert "prediction" in response.json()
def test_predict_missing_feature():
    incomplete_input = sample_input.copy()
    incomplete_input.pop("Age")
    response = client.post("/predict", json=incomplete_input)
    assert response.status_code == 422  # Unprocessable Entity
