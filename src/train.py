"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import tempfile
import os


def rebalance(data):
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )
    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    filter_feat = [
        "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )

    col_transf = make_column_transformer(
        (StandardScaler(), num_cols),
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train_trans = col_transf.fit_transform(X_train)
    X_train_df = pd.DataFrame(X_train_trans, columns=col_transf.get_feature_names_out())

    X_test_trans = col_transf.transform(X_test)
    X_test_df = pd.DataFrame(X_test_trans, columns=col_transf.get_feature_names_out())

    # Log the transformer as an artifact
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "column_transformer.pkl")
        pd.to_pickle(col_transf, path)
        mlflow.log_artifact(path, artifact_path="transformer")

    return col_transf, X_train_df, X_test_df, y_train, y_test


def train(X_train, y_train):
    log_reg = SVC(kernel='rbf')
    log_reg.fit(X_train, y_train)

    # Infer signature for input-output schema
    signature = infer_signature(X_train, log_reg.predict(X_train))

    # Log the model
    mlflow.sklearn.log_model(
        sk_model=log_reg,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(5),
    )

    # Log model hyperparameter
    # mlflow.log_param("max_iter", 1000)

    return log_reg


def main():
    # Set the tracking URI (you can set to a remote server or local file path)
    mlflow.set_tracking_uri("https://6aa7-34-73-229-15.ngrok-free.app")  # or leave default
    mlflow.set_experiment("Bank Churn Prediction")
    

    with mlflow.start_run(run_name="SVM with rbf "):
        df = pd.read_csv("/content/bank-dataset/Churn_Modelling.csv")

        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

        model = train(X_train, y_train)
        y_pred = model.predict(X_test)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

        # Log tag
        mlflow.set_tag("model_type", "SVM classifier")
        mlflow.set_tag("developer", "Omar abbas")

        # Confusion matrix
        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=model.classes_
        )
        conf_mat_disp.plot()

        # Save and log confusion matrix plot as image
        with tempfile.TemporaryDirectory() as tmp_dir:
            fig_path = os.path.join(tmp_dir, "confusion_matrix.png")
            plt.savefig(fig_path)
            mlflow.log_artifact(fig_path, artifact_path="plots")

        plt.show()


if __name__ == "__main__":
    main()
