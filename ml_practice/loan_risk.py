
"""
Title: Supervised Classification Pipeline for Loan Default Risk

Description:
A modular Python exercise script for loading clean loan applicant data,
encoding categorical variables, scaling features, training a Logistic
Regression classifier, evaluating performance, and persisting artifacts.
"""

# ==========================================
# STEP 0: IMPORT LIBRARIES
# ==========================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# PHASE 1: DISCOVERY & TARGET IDENTIFICATION
# ==========================================

def phase_1_discovery_and_target_identification():

    print("\n[PHASE 1] Discovery & Target Identification")
    print("-" * 65)

    # Load dataset
    df = pd.read_csv("loan_risk_data.csv")

    print("\nDataset Shape:", df.shape)
    print("\nDataset Information:")
    print(df.info())

    print("\nFirst Five Records:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Records:", df.duplicated().sum())

    # Separate target and predictors
    y = df["default_flag"]

    # Drop target and non-predictive ID
    X = df.drop(
        columns=["default_flag", "applicant_id"]
    )

    print("\nPredictor Features:")
    print(X.columns.tolist())

    print("\nTarget Distribution:")
    print(y.value_counts())

    print("\nTarget Distribution Percentage:")
    print(y.value_counts(normalize=True) * 100)

    return X, y


# ==========================================
# PHASE 2: FEATURE ENGINEERING & ENCODING
# ==========================================

def phase_2_feature_engineering_and_encoding(X):

    print("\n[PHASE 2] Feature Engineering & Categorical Encoding")
    print("-" * 65)

    # Create a copy to avoid modifying original dataframe
    X = X.copy()

    # Feature Engineering
    if "loan_amount_k" in X.columns and "annual_income_k" in X.columns:

        X["loan_to_income_ratio"] = (
            X["loan_amount_k"] /
            X["annual_income_k"].replace(0, np.nan)
        )

        # Replace infinite and missing ratio values
        X["loan_to_income_ratio"] = (
            X["loan_to_income_ratio"]
            .replace([np.inf, -np.inf], np.nan)
        )

        X["loan_to_income_ratio"] = (
            X["loan_to_income_ratio"]
            .fillna(X["loan_to_income_ratio"].median())
        )

    # Identify categorical columns
    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("\nCategorical Columns:", categorical_columns)

    # One-hot encoding
    X_encoded = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=True,
        dtype=int
    )

    print("\nEncoded Dataset Shape:", X_encoded.shape)

    print("\nEncoded Feature Names:")
    print(X_encoded.columns.tolist())

    print("\nTransformed Dataset:")
    print(X_encoded.head())

    return X_encoded


# ==========================================
# PHASE 3: TRAIN TEST SPLIT & SCALING
# ==========================================

def phase_3_scaling_and_train_test_split(X, y):

    print("\n[PHASE 3] Data Splitting & Feature Scaling")
    print("-" * 65)

    # Stratified splitting preserves class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining Data Shape:", X_train.shape)
    print("Testing Data Shape:", X_test.shape)

    print("\nTraining Target Distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nTesting Target Distribution:")
    print(y_test.value_counts(normalize=True))

    # Initialize scaler
    scaler = StandardScaler()

    # Fit ONLY on training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform testing data using training statistics
    X_test_scaled = scaler.transform(X_test)

    print("\nFeature Scaling Completed Successfully!")

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    )


# ==========================================
# PHASE 4: MODEL TRAINING & EVALUATION
# ==========================================

def phase_4_model_training_and_testing(
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
):

    print("\n[PHASE 4] Model Training & Test Set Evaluation")
    print("-" * 65)

    # Instantiate Logistic Regression
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight="balanced"
    )

    # Train model
    model.fit(X_train_scaled, y_train)

    print("\nModel Training Completed!")

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    print("\nMODEL PERFORMANCE")
    print("-" * 40)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    return model


# ==========================================
# PHASE 5: MODEL PERSISTENCE & INFERENCE
# ==========================================

def phase_5_joblib_model_persistence(
    model,
    scaler,
    feature_names
):

    print("\n[PHASE 5] Model Persistence with Joblib")
    print("-" * 65)

    # Save model, scaler and feature names together
    artifacts = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names
    }

    joblib.dump(
        artifacts,
        "loan_default_pipeline.joblib"
    )

    print("\nModel artifacts saved successfully!")

    # Load saved artifacts
    loaded_artifacts = joblib.load(
        "loan_default_pipeline.joblib"
    )

    loaded_model = loaded_artifacts["model"]
    loaded_scaler = loaded_artifacts["scaler"]
    loaded_features = loaded_artifacts["feature_names"]

    print("\nArtifacts Loaded Successfully!")

    # Create a sample applicant using the training feature structure
    sample_applicant = pd.DataFrame(
        [np.zeros(len(loaded_features))],
        columns=loaded_features
    )

    # Scale sample applicant
    sample_scaled = loaded_scaler.transform(
        sample_applicant
    )

    # Predict
    prediction = loaded_model.predict(sample_scaled)

    probability = loaded_model.predict_proba(sample_scaled)

    print("\nSAMPLE APPLICANT PREDICTION")
    print("-" * 40)

    print(
        "Prediction:",
        "Default Risk" if prediction[0] == 1 else "No Default Risk"
    )

    print(
        f"Default Probability: {probability[0][1]:.4f}"
    )


# ==========================================
# FACADE FUNCTION / ORCHESTRATOR
# ==========================================

def run_ml_pipeline():

    print("=" * 65)
    print("STARTING LOAN RISK CLASSIFICATION LIFECYCLE")
    print("=" * 65)

    # Phase 1
    X, y = phase_1_discovery_and_target_identification()

    # Phase 2
    X_encoded = phase_2_feature_engineering_and_encoding(X)

    # Phase 3
    (
        X_tr_scaled,
        X_te_scaled,
        y_tr,
        y_te,
        scaler
    ) = phase_3_scaling_and_train_test_split(
        X_encoded,
        y
    )

    # Phase 4
    model = phase_4_model_training_and_testing(
        X_tr_scaled,
        X_te_scaled,
        y_tr,
        y_te
    )

    # Phase 5
    phase_5_joblib_model_persistence(
        model,
        scaler,
        X_encoded.columns.tolist()
    )

    print("\n" + "=" * 65)
    print("LOAN RISK CLASSIFICATION LIFECYCLE COMPLETED")
    print("=" * 65)


# ==========================================
# MAIN GUARD
# ==========================================

if __name__ == "__main__":
    run_ml_pipeline()
