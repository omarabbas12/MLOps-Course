# Bank Churn Prediction 🏦

This repository presents an ML pipeline developed to predict customer churn in a bank using several classification models. The project includes proper experiment tracking using **MLflow**, providing visibility into model metrics and comparisons.

---

## 🔍 Problem Statement

Bank churn prediction is a classification problem that aims to identify customers who are likely to leave the bank. Understanding this behavior helps the bank improve retention strategies.

---

## 📁 Project Structure

.
├── train.py # Core training script
├── data/ # Dataset (cleaned/preprocessed)
├── models/ # Saved models (optional)
├── mlruns/ # MLflow tracking artifacts
├── README.md # This file
└── requirements.txt # Dependencies

---

## ✅ Models Considered

The following classification models were trained and evaluated:

- **Support Vector Machine (SVM)** with linear and RBF kernels  
- **Decision Tree Classifier**  
- **Random Forest Classifier**

Each model was logged and tracked using **MLflow**, including key metrics such as:

- Accuracy
- F1 Score
- Precision
- Recall

---

## 📊 MLflow Visualization

We used MLflow to track and visualize the performance of each model across multiple metrics.

### Example Runs:

| Run Name            | Accuracy | F1 Score | Precision | Recall |
|---------------------|----------|----------|-----------|--------|
| SVM with RBF        | 0.7629   | 0.7526   | 0.7723    | 0.7338 |
| Random Forest       | 0.7653   | 0.7532   | 0.7794    | 0.7288 |
| Decision Tree       | 0.6934   | 0.6903   | 0.6852    | 0.6955 |
| SVM (Linear)        | 0.7629   | 0.7526   | 0.7723    | 0.7338 |
| sedate-chimp-339    | 0.7065   | 0.6955   | 0.7093    | 0.6822 |

### MLflow Screenshots

#### MLflow Runs Table

![MLflow Runs Table](./ML%20flow%20.png)

#### MLflow Metrics Charts

![MLflow Metrics](./mlflow%20with%20visualization.png)

---

## 📌 Key Takeaways

- **Random Forest** slightly outperformed the others in accuracy and precision.
- All models were evaluated on the same dataset using consistent metrics.
- Experiment tracking with **MLflow** enabled efficient comparison and reproducibility.

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/bank-churn-prediction.git
   cd bank-churn-prediction

