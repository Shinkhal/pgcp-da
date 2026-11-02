"""
Title: Healthcare Classification Lifecycle
Description: Module 7 Supervised Segmentation & Classification Exercise.

"""


# Step 0: Import necessary libraries
# TODO: Import pandas, LogisticRegression from sklearn.linear_model, accuracy_score, and confusion_matrix
import pandas as pd
import numpy as np
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery & Data Setup
    - Goal: Load 'patient_classification_data.csv' and inspect class distribution.
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)
    # TODO: Load CSV file using pandas
    # TODO: Check class balance of 'High_Risk_Hypertension' (value_counts)

    df = pd.read_csv("patient_classification_data.csv")
    df.columns = ( df.columns.str.strip().str.lower().str.replace(' ','_'))
    df.info()
    print(df['high_risk_hypertension'].value_counts())
    print(df.head())
    return df

def phase_2_supervised_segmentation(df):
    """
    [Phase 2] Supervised Segmentation & Feature Selection
    - Goal: Analyze feature differences between risk groups.
    """
    print("\n[PHASE 2] Supervised Segmentation")
    print("-" * 65)
    # TODO: Group data by 'High_Risk_Hypertension' and calculate mean of numeric features (BMI, Daily_Steps)

    numeric_cols = ['bmi', 'daily_steps']

    group_analysis = df.groupby(
        'high_risk_hypertension'
    )[numeric_cols].mean()

    print(group_analysis)

    return df



def phase_3_classification_modeling(df):
    """
    [Phase 3] Classification Model Induction
    - Goal: Train a Logistic Regression model to classify high-risk patients.
    """
    print("\n[PHASE 3] Classification Modeling")
    print("-" * 65)
    # TODO: Define features X = df[['BMI', 'Daily_Steps']] and target y = df['High_Risk_Hypertension']
    # TODO: Fit LogisticRegression model


    X = df[['bmi', 'daily_steps']]
    y = df['high_risk_hypertension']

    model = LogisticRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_

    print(f"Regression Equation: y = {slope[0]:.4f} * BMI + {slope[1]:.4f} * Daily_Steps + {intercept[0]:.4f}")
    return model, X, y


def phase_4_model_evaluation(model, X, y):
    """
    [Phase 4] Evaluating Classifiers & Generalization
    - Goal: Compute Accuracy and Confusion Matrix.
    - Provide clinical prescriptive actions.
    """
    print("\n[PHASE 4] Model Evaluation & Clinical Action")
    print("-" * 65)
    # TODO: Generate predictions and calculate accuracy_score and confusion_matrix
    # TODO: Print prescriptive clinical intervention recommendations

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)

    print(f"Model Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    if accuracy < 0.8:
        print("Clinical Recommendation: Model accuracy is below threshold. Consider collecting more data or feature engineering.")
    else:
        print("Clinical Recommendation: Model is performing well. Use predictions to prioritize high-risk patients for intervention.")


def run_analytics_lifecycle():
    """
    Facade Orchestrator function.
    """
    print("=" * 65)
    print("STARTING HEALTHCARE CLASSIFICATION LIFECYCLE ")
    print("=" * 65)

    df = phase_1_discovery_and_setup()
    df = phase_2_supervised_segmentation(df)
    model, X, y = phase_3_classification_modeling(df)
    phase_4_model_evaluation(model, X, y)
    print(df)

if __name__ == "__main__":
    run_analytics_lifecycle()