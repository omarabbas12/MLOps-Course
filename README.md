# Bank Churn Prediction 🏦 + FastAPI Deployment

This repository presents a full ML pipeline to predict customer churn for a bank, complete with:

- ✅ Model training and tracking using **MLflow**
- ✅ Deployment using **FastAPI**
- ✅ Automated testing with **pytest**
- ✅ Interactive API documentation via **Swagger UI**

---

## 🔍 Problem Statement

Bank churn prediction is a classification problem aimed at identifying customers likely to leave the bank. Early identification helps improve customer retention through better service and offers.

---

## ✅ Models Considered

The following classification models were trained and evaluated:

- **Support Vector Machine (SVM)** (Linear & RBF)
- **Decision Tree Classifier**
- **Random Forest Classifier**

Each model was tracked using **MLflow**, logging:

- Accuracy
- F1 Score
- Precision
- Recall

---

## 📊 MLflow Tracking

We used MLflow to track and compare model runs with consistent metrics.

### Sample Results:

| Run Name            | Accuracy | F1 Score | Precision | Recall |
|---------------------|----------|----------|-----------|--------|
| SVM with RBF        | 0.7629   | 0.7526   | 0.7723    | 0.7338 |
| Random Forest       | 0.7653   | 0.7532   | 0.7794    | 0.7288 |
| Decision Tree       | 0.6934   | 0.6903   | 0.6852    | 0.6955 |

> ✅ The best-performing model was saved using `joblib` for later deployment.

---

## 🖥️ FastAPI Deployment

The trained model and `ColumnTransformer` were deployed using **FastAPI**, exposing endpoints:

### Available Endpoints

| Method | Route        | Description                     |
|--------|--------------|---------------------------------|
| GET    | `/`          | Health message                  |
| GET    | `/health`    | Status check                    |
| POST   | `/predict`   | Predict churn from user input   |

### Input Format (`/predict`):

```json
{
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
```

### Output Example:

```json
{
  "prediction": 0
}
```

> `0` means the customer is predicted **not to churn**, `1` means they **will churn**.

---

## ⚙️ Testing with Pytest

Automated unit tests are provided in `test_api.py` and include:

- `/` (GET): Home
- `/health` (GET): Health check
- `/predict` (POST): Valid input
- `/predict`: All-zero edge input
- `/predict`: Missing input (should fail)

Run tests:

```bash
pytest
```

---

## 🧪 Swagger Documentation

Once FastAPI is running:

📄 Access interactive API docs at:

```
http://127.0.0.1:8000/docs
```

Use the "Try it out" button to POST sample inputs.

---

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/bank-churn-prediction.git
cd bank-churn-prediction
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # on Linux/Mac
venv\Scripts\activate     # on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python main.py
```

---

## 📁 File Structure Overview

```
MLOps-Course/
│
├── model.pkl                     # Trained model
├── column_transformer.pkl        # Preprocessing pipeline
├── main.py                       # FastAPI app
├── test_api.py                   # Pytest test suite
├── logs/                         # Logs for app usage
├── requirements.txt              # Dependencies
└── README.md
```

---

## 📌 Notes

- Make sure `model.pkl` and `column_transformer.pkl` are present in the project root or update the path accordingly in `main.py`.
- Models were trained with `scikit-learn==1.6.1` and NumPy `1.x` — avoid NumPy 2.x to prevent compatibility issues.

---

## 📸 Screenshots

![MLflow UI](https://github.com/user-attachments/assets/fca6dd6e-c4d7-4975-a38c-f53b49590f3c)

---

## 🛠️ Future Improvements

- Add frontend UI (e.g. Streamlit)
- Deploy with Docker or Render
- Monitor model drift and retrain

---

## 📬 Contact

For questions, contact [yourname@domain.com](mailto:yourname@domain.com) or open an issue.
