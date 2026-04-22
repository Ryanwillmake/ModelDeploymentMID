import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import numpy as np
from pipeline import load_data, preprocess, cls_pipeline, reg_pipeline

mlflow.set_experiment("MD_Placement_Prediction")

df = load_data("B.csv")
X, y_cls, y_reg = preprocess(df)

X_train, X_test, y_cls_train, y_cls_test = train_test_split(X, y_cls, test_size=0.2, random_state=42)
_, _, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="classification"):
    cls_pipeline.fit(X_train, y_cls_train)
    y_pred_cls = cls_pipeline.predict(X_test)
    acc = accuracy_score(y_cls_test, y_pred_cls)
    f1 = f1_score(y_cls_test, y_pred_cls)
    mlflow.log_param("model", "GradientBoostingClassifier")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(cls_pipeline, "cls_model")
    print(f"Klasifikasi — Accuracy: {acc:.4f} | F1: {f1:.4f}")

with mlflow.start_run(run_name="regression"):
    reg_pipeline.fit(X_train, y_reg_train)
    y_pred_reg = reg_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_reg))
    r2 = r2_score(y_reg_test, y_pred_reg)
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)
    mlflow.sklearn.log_model(reg_pipeline, "reg_model")
    print(f"Regresi — RMSE: {rmse:.4f} | R2: {r2:.4f}")

joblib.dump(cls_pipeline, "model_klasifikasi.pkl")
joblib.dump(reg_pipeline, "model_regresi.pkl")
print("Model disimpan: model_klasifikasi.pkl & model_regresi.pkl")