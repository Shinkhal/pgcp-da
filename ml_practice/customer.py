"""
Title: Supervised Classification Pipeline with Random Forest
Description: A modular Python exercise script for loading clean customer churn data,
             identifying target/predictors, encoding categorical variables,
             scaling features, training a Random Forest classifier, evaluating performance,
             and persisting the trained pipeline artifacts using joblib.
"""
import joblib
# Step 0: Import necessary libraries
# TODO: Import pandas, numpy, train_test_split, StandardScaler,
#       LogisticRegression, accuracy_score, classification_report, confusion_matrix, and joblib

import pandas as pd
import numpy as np
from matplotlib import artist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, f1_score, \
    precision_score


def phase_1_discovery_and_target_identification():
    """
    [Phase 1] Discovery, Setup & Target/Predictor Identification
    - Goal: Load 'customer_churn_data.csv' and inspect structure.
    - Drop non-predictive identifiers (e.g., 'customer_id').
    - Identify Target feature (y: 'churn_flag') and Predictor features (X).
    """
    print("\n[PHASE 1] Discovery & Target Identification")
    print("-" * 65)
    # TODO: Load CSV file using pandas
    # TODO: Separate target feature y ('churn_flag') and predictor features X
    # TODO: Drop ID column ('customer_id')
    # TODO: Inspect class distribution of target variable

    df = pd.read_csv('customer_churn_data.csv')
    print(df.shape)
    df.info()

    print(df.head())
    print(f'Missing values:\n{df.isnull().sum()}')
    print(f'Duplicate records: {df.duplicated().sum()}')

    y = df['churn_flag']
    X = df.drop(columns=['customer_id', 'churn_flag'])

    print(f'Target class distribution:\n{y.value_counts(normalize=True)}')

    return X, y


def phase_2_feature_engineering_and_encoding(X):
    """
    [Phase 2] Categorical Encoding & Feature Engineering
    - Goal: Encode categorical variables (e.g., 'contract_type', 'internet_service').
    - Optionally create engineered interaction features (e.g., charges per tenure month).
    """
    print("\n[PHASE 2] Feature Engineering & Categorical Encoding")
    print("-" * 65)
    # TODO: Apply one-hot encoding to categorical columns using pd.get_dummies()
    # TODO: Add engineered features if desired (e.g., X['avg_monthly_spend'] = X['total_charges'] / X['tenure_months'])
    # TODO: Display head of transformed feature matrix


    X['average_monthly_spend'] = (
        X['total_charges'] / X['tenure_months'].replace(0, np.nan)
    )

    X['average_monthly_spend'] = (
        X['average_monthly_spend'].replace([np.inf, -np.inf], np.nan)
    )
    X['average_monthly_spend'] = X['average_monthly_spend'].fillna(X['average_monthly_spend'].median())

    categorical_cols = X.select_dtypes(include=['object','category','bool']).columns

    x_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)

    print(f"Encoded feature matrix shape: {x_encoded.shape}")

    print(f"Feature matrix after engineering shape: {x_encoded.shape}")
    print(x_encoded.head())
    return x_encoded



def phase_3_scaling_and_train_test_split(X, y):
    """
    [Phase 3] Train-Test Split & Feature Scaling
    - Goal: Split dataset into Training and Testing sets.
    - Fit StandardScaler on training data and transform both train and test sets.
    """
    print("\n[PHASE 3] Data Splitting & Feature Scaling")
    print("-" * 65)
    # TODO: Perform train_test_split (test_size=0.2, random_state=42)
    # TODO: Initialize StandardScaler and scale numerical features
    # TODO: Fit scaler on X_train, transform X_train and X_test

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set shape: {X_train_scaled.shape}, Testing set shape: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# def phase_4_model_training_and_testing(X_train_scaled, X_test_scaled, y_train, y_test):
#     """
#     [Phase 4] LogisticRegression & Classification Metrics
#     - Goal: Train a Logistic Regression model and evaluate on test set.
#     - Predict on test set and calculate evaluation metrics (Accuracy, Precision, Recall, F1).
#     """
#     print("\n[PHASE 4] Model Training & Test Set Evaluation")
#     print("-" * 65)
#     # TODO: Instantiate LogisticRegrssion and fit on training data
#     # TODO: Generate predictions on test set
#     # TODO: Compute accuracy_score and classification_report
#
#     model = LogisticRegression(max_iter=1000, random_state=42)
#     model.fit(X_train_scaled, y_train)
#     slope = model.coef_[0]
#     intercept = model.intercept_
#     print(f"Model coefficients: {slope}, Intercept: {intercept}")
#
#     y_pred = model.predict(X_test_scaled)
#
#     print("Model Prediction Completed. Evaluation Metrics:")
#     accuracy = accuracy_score(y_test,y_pred)
#     print(f"Accuracy: {accuracy:.4f}")
#     precision = precision_score(y_test,y_pred)
#     print(f"Precision: {precision:.4f}")
#     recall = recall_score(y_test,y_pred)
#     print(f"Recall: {recall:.4f}")
#     f1 = f1_score(y_test,y_pred)
#     print(f"F1 Score: {f1:.4f}")
#     cm = confusion_matrix(y_test,y_pred)
#     print("Confusion Matrix:")
#     print(cm)

from sklearn.ensemble import RandomForestClassifier

def phase_4_model_training_and_testing(
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
):

    print("\n[PHASE 4] Random Forest Model Training & Evaluation")
    print("-" * 65)

    # Initialize Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    )

    # Train model
    model.fit(X_train_scaled, y_train)

    print("\nModel Training Completed!")

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluation metrics
    print("\nMODEL PERFORMANCE")
    print("-" * 40)

    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred, zero_division=0):.4f}")

    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

    # Feature Importance
    importance = pd.DataFrame({
        "Feature": X_train_scaled.shape[1] and range(X_train_scaled.shape[1]),
        "Importance": model.feature_importances_
    })

    print("\nFeature Importances:")
    print(importance.sort_values(
        by="Importance",
        ascending=False
    ))

    return model


def phase_5_joblib_model_persistence(model, scaler, feature_names):
    """
    [Phase 5] Model Persistence & Future Deployment Setup
    - Goal: Save the trained model, fitted scaler, and feature list using joblib.
    - Test loading saved artifacts to run an inference on a new sample.
    """
    print("\n[PHASE 5] Model Persistence with Joblib")
    print("-" * 65)
    # TODO: Use joblib.dump() to save the model and scaler to disk (.joblib files)
    # TODO: Load saved artifacts using joblib.load() and perform sample inference
    artifacts = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names
    }
    joblib.dump(artifacts, "customer_churn_model.joblib")
    print("Model saved successfully")
    loaded = joblib.load("customer_churn_model.joblib")
    loaded_model = loaded["model"]
    loaded_scaler = loaded["scaler"]
    loaded_features = loaded["feature_names"]

    test_data = pd.DataFrame(
        [[5000, 45.0, 150, 3.5, 20, 1, 0, 1]],
        columns=loaded_features
    )
    test_data_scaled = loaded_scaler.transform(test_data)
    predicted_class = loaded_model.predict(test_data_scaled)[0]
    print(f"Predicted class for new sample: {predicted_class}")


# ==========================================
# THE FACADE FUNCTION (Orchestrator)
# ==========================================

def run_ml_pipeline():
    """
    Facade function coordinating the entire classification pipeline.
    """
    print("=" * 65)
    print("STARTING SUPERVISED CLASSIFICATION LIFECYCLE (Random Forest)")
    print("=" * 65)

    # Executing phases chronologically
    X, y = phase_1_discovery_and_target_identification()
    X_encoded = phase_2_feature_engineering_and_encoding(X)
    X_tr_scaled, X_te_scaled, y_tr, y_te, scaler = phase_3_scaling_and_train_test_split(X_encoded, y)
    model = phase_4_model_training_and_testing(X_tr_scaled, X_te_scaled, y_tr, y_te)
    phase_5_joblib_model_persistence(model, scaler, X_encoded.columns.tolist())


# ==========================================
# MAIN GUARD
# ==========================================
if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    run_ml_pipeline()