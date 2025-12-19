import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.utils import estimator_html_repr
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

import dagshub

dagshub.init(repo_owner='zickrian', repo_name='MSML', mlflow=True)

# --- SETUP MLflow ---
mlflow.set_tracking_uri("http://127.0.0.1:5000")  
mlflow.set_tracking_uri("https://dagshub.com/zickrian/MSML.mlflow")
mlflow.set_experiment("Diabetes_RF_Tuned_Models")

# --- LOAD DATA ---
data_path = 'diabetes_preprosesing.csv'
df = pd.read_csv(data_path)
X = df.drop(columns=['Outcome'])
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, stratify=y, random_state=42
)

# --- SMOTE ---
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# --- Grid Search Parameters ---
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
grid.fit(X_train_resampled, y_train_resampled)

# --- MLflow Manual Logging ---
with mlflow.start_run(run_name="RandomForest_Tuning_GridSearch_Manual_Logging"):
	best_model = grid.best_estimator_

	y_pred = best_model.predict(X_test)
	y_pred_proba = best_model.predict_proba(X_test)[:, 1]

	acc = accuracy_score(y_test, y_pred)
	prec = precision_score(y_test, y_pred)
	rec = recall_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred)

	# Manual Logging - Parameters
	mlflow.log_param("best_n_estimators", best_model.n_estimators)
	mlflow.log_param("best_max_depth", best_model.max_depth)
	mlflow.log_param("best_min_samples_split", best_model.min_samples_split)
	mlflow.log_param("best_min_samples_leaf", best_model.min_samples_leaf)
	mlflow.log_param("best_bootstrap", best_model.bootstrap)
	mlflow.log_param("cv_folds", 3)
	mlflow.log_param("test_size", 0.2)
	mlflow.log_param("smote_applied", True)
	mlflow.log_param("random_state", 42)

	# Manual Logging - Metrics
	mlflow.log_metric("accuracy", acc)
	mlflow.log_metric("precision", prec)
	mlflow.log_metric("recall", rec)
	mlflow.log_metric("f1_score", f1)
	mlflow.log_metric("best_cv_score", grid.best_score_)

	# Manual Logging - Model with proper parameters
	input_example = X_test.iloc[:5]
	mlflow.sklearn.log_model(
		best_model,
		name="model",
		input_example=input_example,
		registered_model_name="RandomForest_Diabetes_Best"
	)

	# Ensure model folder is visible under Artifacts by saving & logging
	model_local_dir = os.path.join("artifacts", "model")
	import shutil
	if os.path.exists(model_local_dir):
		shutil.rmtree(model_local_dir)
	os.makedirs(model_local_dir, exist_ok=True)
	mlflow.sklearn.save_model(best_model, model_local_dir)
	mlflow.log_artifacts(model_local_dir, artifact_path="model")

	# --- Generate Estimator HTML ---
	estimator_html = estimator_html_repr(best_model)
	mlflow.log_text(estimator_html, "estimator.html")

	# --- Generate Confusion Matrix ---
	cm = confusion_matrix(y_test, y_pred)
	fig = plt.figure(figsize=(6,4))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
	plt.title("Confusion Matrix")
	plt.xlabel("Predicted")
	plt.ylabel("Actual")
	plt.tight_layout()
	mlflow.log_figure(fig, "training_confusion_matrix.png")
	plt.close(fig)

	# --- Generate Metric Info JSON ---
	import json
	metric_info = {
		"accuracy": float(acc),
		"precision": float(prec),
		"recall": float(rec),
		"f1_score": float(f1),
		"best_cv_score": float(grid.best_score_)
	}
	mlflow.log_dict(metric_info, "metric_info.json")

	# --- Calculate ROC AUC (for print only, not logged as artifact) ---
	fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
	roc_auc = auc(fpr, tpr)
	mlflow.log_metric("roc_auc", roc_auc)

	print("--- Best Model Evaluation ---")
	print(f"Accuracy : {acc:.4f}")
	print(f"Precision: {prec:.4f}")
	print(f"Recall   : {rec:.4f}")
	print(f"F1 Score : {f1:.4f}")
	print(f"ROC AUC  : {roc_auc:.4f}")
