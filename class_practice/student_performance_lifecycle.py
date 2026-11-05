"""
Title: Student Academic Performance Analytics Lifecycle
Description: Complete the TODO sections across the 5 lifecycle phases.
Dataset: student_performance.csv (70 rows)
"""

# Step 0: Import necessary libraries
# TODO: Import pandas, numpy, LogisticRegression,
#       train_test_split, and StandardScaler.
# TODO: Import accuracy_score, precision_score, recall_score,
#       f1_score, and confusion_matrix.

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)



def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery & Data Setup (5 Marks)
    - Load and inspect the dataset.
    - Handle missing numerical values.
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)

    # TODO: Load CSV
    # TODO: Inspect dataset
    # TODO: Handle missing values
    df = pd.read_csv("student_performance.csv")
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_")
    )
    print(df.head())
    df.info()
    missing = df.isnull().sum()
    missing_percentage = (df.isnull().mean() * 100).round(2)
    missing_report = pd.DataFrame({
        "Missing Values": missing,
        "Missing Percentage": missing_percentage
    })
    print(missing_report)

    missing_cols = df.select_dtypes(include=np.number).columns[df.select_dtypes(include=np.number).isnull().any()]
    for col in missing_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    print("Missing values filled with median for numerical columns.")

    return df


def phase_2_descriptive_statistics(df):
    """
    [Phase 2] Descriptive Statistics (10 Marks)
    - Analyze Exam_Score.
    - Calculate statistical measures.
    - Detect outliers using IQR.
    - Compare performance across streams.
    """
    print("\n[PHASE 2] Descriptive Statistics")
    print("-" * 65)

    # TODO: Calculate statistical measures
    # TODO: Apply IQR method
    # TODO: Identify outliers
    # TODO: Analyze streams

    target = df['exam_score']
    mean = target.mean()
    median = target.median()
    std = target.std()
    q1 = target.quantile(0.25)
    q3 = target.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(target < lower_bound) | (target > upper_bound)]
    print(f"Mean: {mean:.2f}, Median: {median:.2f}, Std Dev: {std:.2f}")
    print(f"Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
    print(f"Outliers:\n{outliers}")

    stream_analysis = df.groupby('stream')['exam_score'].mean()
    print(f"Average Exam_Score per Stream:\n{stream_analysis}")
    return df


def phase_3_bivariate_association(df):
    """
    [Phase 3] Bivariate Association (10 Marks)
    - Examine relationships among student attributes.
    - Analyze correlation with Exam_Score.
    """
    print("\n[PHASE 3] Bivariate Association")
    print("-" * 65)

    # TODO: Compute correlation matrix
    # TODO: Analyze selected correlations
    # TODO: Identify strongest feature

    numeric_cols = df.select_dtypes(include=np.number).columns
    correlation_matrix = df[numeric_cols].corr(method='pearson')
    print("Pearson Correlation Matrix:")
    print(correlation_matrix)
    strongest_feature = correlation_matrix['exam_score'].drop('exam_score').abs().idxmax()
    print(f"Feature most strongly correlated with Exam_Score: {strongest_feature}")
    return df


def phase_4_predictive_modeling(df):

    print("\n[PHASE 4] Predictive Modeling")
    print("-" * 65)

    # Create target
    df['pass_fail'] = np.where(df['exam_score'] >= 50, 1, 0)

    # Prepare features
    X = df[['study_hours', 'attendance_pct', 'sleep_hours', 'stream']]
    y = df['pass_fail']

    # Encode categorical feature
    X = pd.get_dummies(X, columns=['stream'], drop_first=True, dtype=int)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale features
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    print("Model trained successfully.")
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)

    return model, X_test, y_test, y_pred


def phase_5_evaluation_and_reporting(model, X_test, y_test, y_pred, df):
    """
    [Phase 5] Evaluation & Reporting (5 Marks)
    - Evaluate classification performance.
    - Recommend academic interventions.
    """
    print("\n[PHASE 5] Evaluation & Recommendations")
    print("-" * 65)

    # TODO: Calculate classification metrics
    # TODO: Display confusion matrix
    # TODO: Interpret model performance
    # TODO: Generate recommendations

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)


def run_analytics_lifecycle():
    """Facade Orchestrator function."""

    print("=" * 65)
    print("STUDENT PERFORMANCE CLASSIFICATION (Out of 40 Marks)")
    print("=" * 65)

    df = phase_1_discovery_and_setup()
    df = phase_2_descriptive_statistics(df)
    df = phase_3_bivariate_association(df)
    model, X_test, y_test, y_pred = phase_4_predictive_modeling(df)
    phase_5_evaluation_and_reporting(
        model, X_test, y_test, y_pred, df
    )


if __name__ == "__main__":
    run_analytics_lifecycle()