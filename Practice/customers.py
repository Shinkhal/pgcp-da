"""
Title: Customer Churn & Missing Data Analytics Lifecycle

Description:
Complete the TODO sections across the 6 lifecycle phases.

Dataset: customer_churn_mcar_practical.csv

Objective:
Analyze customer churn, investigate MCAR missing values,
perform missing value imputation, feature engineering,
and build a Logistic Regression classification model.

Total Marks: 60
"""

# Step 0: Import necessary libraries
# TODO: Import pandas, numpy, matplotlib, seaborn
# TODO: Import SimpleImputer, OneHotEncoder, StandardScaler
# TODO: Import ColumnTransformer, Pipeline
# TODO: Import train_test_split
# TODO: Import LogisticRegression
# TODO: Import accuracy_score, confusion_matrix,
#       classification_report

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery & Data Setup (10 Marks)

    Goal:
    Load 'customer_churn_mcar_practical.csv' and inspect
    the dataset structure.

    Tasks:
    - Display first 5 records and dataset dimensions.
    - Inspect column data types.
    - Identify missing values and duplicates.
    - Generate a missing value percentage report.
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)

    # TODO: Load CSV using pandas
    # TODO: Display first 5 rows and dataset shape
    # TODO: Display dataset information and descriptive statistics
    # TODO: Identify duplicate records
    # TODO: Calculate missing values per column
    # TODO: Calculate missing value percentage per column
    # TODO: Display only columns containing missing values

    df = pd.read_csv("customer_churn_mcar_practical.csv")
    print(df.head())
    print(f"Dataset shape: {df.shape}")
    df.info()
    print(df.describe())

    duplicates = df.duplicated().sum()
    print(f"Number of duplicate records: {duplicates}")

    missing = df.isnull().sum()
    missing_percentage = (missing / len(df)) * 100
    missing_report = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_percentage
    })
    print("Missing Value Report:")
    print(missing_report[missing_report['Missing Values'] > 0])


    return df


def phase_2_mcar_analysis(df):
    """
    [Phase 2] Missing Data Analysis & MCAR Investigation
    """
    print("\n[PHASE 2] MCAR Analysis")
    print("-" * 65)

    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=np.number).columns

    categorical_cols = df.select_dtypes(
        include=['object', 'category', 'string']
    ).columns

    print("Numerical Columns:", list(numerical_cols))
    print("Categorical Columns:", list(categorical_cols))

    # Missing value report
    missing = df.isnull().sum()
    missing_percentage = (missing / len(df)) * 100

    missing_report = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_percentage
    })

    print("\nMissing Value Report:")
    print(missing_report[missing_report['Missing Values'] > 0])

    # Visualize missing values BEFORE imputation
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Value Pattern")
    plt.xlabel("Features")
    plt.ylabel("Records")
    plt.show()

    # Create missingness indicators
    missing_indicators = df.isnull().astype(int)

    print("\nMissingness Indicator Summary:")
    print(missing_indicators.sum()[missing_indicators.sum() > 0])

    # Explain missing data mechanisms
    print("\nMissing Data Mechanisms:")
    print("MCAR: Missingness is independent of observed and missing values.")
    print("MAR: Missingness depends on other observed variables.")
    print("MNAR: Missingness depends on the missing value itself.")

    return df

def phase_3_missing_value_imputation(df):
    """
    [Phase 3] Missing Value Treatment (10 Marks)

    Goal:
    Handle missing values in numerical and categorical columns.

    Tasks:
    - Use mean or median imputation for numerical features.
    - Use mode imputation for categorical features.
    - Do not blindly impute customer_id or churn.
    - Verify that missing values have been handled.
    """
    print("\n[PHASE 3] Missing Value Imputation")
    print("-" * 65)

    # TODO: Identify numerical and categorical features
    # TODO: Calculate skewness of numerical features
    # TODO: Select suitable imputation strategies
    # TODO: Impute numerical columns using mean or median
    # TODO: Impute categorical columns using mode
    # TODO: Verify remaining missing values
    # TODO: Return cleaned dataset

    numerical_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(
        include=['object', 'category', 'string']
    ).columns

    # Impute numerical columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            if df[col].skew() < 1:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)

    # Impute categorical columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0])
        else:
            continue

    missing = df.isnull().sum()
    print("\nRemaining Missing Values After Imputation:")
    print(missing[missing > 0])


    return df


def phase_4_feature_engineering(df):
    """
    [Phase 4] Feature Engineering & Transformation (10 Marks)

    Goal:
    Prepare the dataset for Logistic Regression.

    Tasks:
    - Separate independent variables and target variable.
    - Remove customer_id from predictive features.
    - Apply One-Hot Encoding to categorical features.
    - Apply StandardScaler to numerical features.
    - Use ColumnTransformer to combine transformations.

    Important:
    Prevent data leakage by fitting transformations
    only on training data.
    """
    print("\n[PHASE 4] Feature Engineering")
    print("-" * 65)

    # TODO: Separate X and y

    # TODO: Remove customer_id from X

    # TODO: Identify numerical and categorical features

    # TODO: Create numerical preprocessing pipeline

    # TODO: Create categorical preprocessing pipeline

    # TODO: Use ColumnTransformer to combine both pipelines

    # TODO: Return X, y, and preprocessor

    pass


def phase_5_predictive_modeling(X, y, preprocessor):
    """
    [Phase 5] Predictive Modeling & Classification (10 Marks)

    Goal:
    Predict whether a customer will churn using Logistic Regression.

    Tasks:
    - Split data into training and testing sets.
    - Use 80% training data and 20% testing data.
    - Use stratified sampling.
    - Build a Pipeline containing preprocessing and Logistic Regression.
    - Train the model and generate predictions.
    """
    print("\n[PHASE 5] Predictive Modeling")
    print("-" * 65)

    # TODO: Perform train_test_split

    # TODO: Use random_state=42 and stratify=y

    # TODO: Create a Pipeline with:
    #       1. Preprocessor
    #       2. LogisticRegression

    # TODO: Fit the model using training data

    # TODO: Generate predictions on test data

    # TODO: Return trained model and predictions

    pass


def phase_6_evaluation_and_reporting(model, X_test, y_test, y_pred, df):
    """
    [Phase 6] Model Evaluation & Business Recommendations (10 Marks)

    Goal:
    Evaluate classification performance and derive
    business insights from customer churn patterns.

    Tasks:
    - Calculate accuracy, precision, recall, and F1-score.
    - Generate confusion matrix.
    - Interpret false positives and false negatives.
    - Calculate overall churn percentage.
    - Compare churn across plan and contract types.
    - Provide business recommendations.
    """
    print("\n[PHASE 6] Evaluation & Business Recommendations")
    print("-" * 65)

    # TODO: Calculate accuracy score

    # TODO: Generate confusion matrix

    # TODO: Generate classification report

    # TODO: Interpret True Positive, True Negative,
    #       False Positive, and False Negative

    # TODO: Calculate customer churn percentage

    # TODO: Analyze churn by plan_type

    # TODO: Analyze churn by contract_type

    # TODO: Provide business recommendations

    pass


def run_analytics_lifecycle():
    """
    Facade Orchestrator function.
    Executes all six phases chronologically.
    """
    print("=" * 65)
    print("STARTING CUSTOMER CHURN ANALYTICS LIFECYCLE")
    print("TOTAL MARKS: 60")
    print("=" * 65)

    # Executing phases chronologically

    df = phase_1_discovery_and_setup()

    df = phase_2_mcar_analysis(df)

    df = phase_3_missing_value_imputation(df)

    # X, y, preprocessor = phase_4_feature_engineering(df)

    # model, y_pred = phase_5_predictive_modeling(
    #     X, y, preprocessor
    # )

    # phase_6_evaluation_and_reporting(
    #     model, X_test, y_test, y_pred, df
    # )


if __name__ == "__main__":
    run_analytics_lifecycle()