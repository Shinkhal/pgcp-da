"""
Title: Healthcare Patient Analytics Lifecycle
Description: Fill in the TODO sections across the 5 lifecycle phases using
             'patient_health_data.csv'.

"""


# Step 0: Import necessary libraries
# TODO: Import pandas, numpy, LinearRegression from sklearn.linear_model, and evaluation metrics
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery & Data Setup
    - Goal: Load 'patient_health_data.csv' and inspect structure.
    - Check for null values and print summary info.
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)
    # TODO: Load CSV file using pandas
    # TODO: Print first 5 rows and verify dataset shape
    # TODO: Check for any missing values


    df = pd.read_csv("patient_health_data.csv")

    print(f"Dataset shape: {df.shape}")
    print("Information about the dataset:")
    df.info()
    print("Missing values in each column:")
    print(df.isnull().sum())
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_")
    )
    print("First 5 rows of the dataset:")
    print(df.head())

    return df


def phase_2_descriptive_statistics(df):
    """
    [Phase 2] Descriptive Statistics & Data Measures
    - Goal: Compute measures of central tendency and dispersion
      for 'Systolic_BP'.
    - Check for potential outliers using the IQR method.
    """
    print("\n[PHASE 2] Descriptive Statistics ")
    print("-" * 65)
    # TODO: Calculate Mean, Median, Std Dev, Variance, and IQR for Systolic_BP
    # TODO: Identify any values falling outside 1.5 * IQR bounds

    mean = df['systolic_bp'].mean()
    median = df['systolic_bp'].median()
    std_dev = df['systolic_bp'].std()
    variance = df['systolic_bp'].var()
    Q1 = df['systolic_bp'].quantile(0.25)
    Q3 = df['systolic_bp'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"Mean: {mean}, Median: {median}, Std Dev: {std_dev}, Variance: {variance}")

    print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
    outliers = df[(df['systolic_bp'] < lower_bound) | (df['systolic_bp'] > upper_bound)]
    print(f"Outliers detected:\n{outliers}")

    return df



def phase_3_bivariate_association(df):
    """
    [Phase 3] Bivariate Association & Correlation
    - Goal: Examine relationships between BMI, Daily Steps, and Systolic BP.
    - Compute Pearson correlation matrix.
    """
    print("\n[PHASE 3] Bivariate Association ")
    print("-" * 65)
    # TODO: Select numeric columns and compute Pearson correlation matrix
    # TODO: Highlight correlations involving Systolic_BP

    numeric_cols = df.select_dtypes(include=['number']).columns
    correlation_matrix = df[numeric_cols].corr(method='pearson')
    print("Pearson Correlation Matrix:")
    print(correlation_matrix)
    #
    # correlation = correlation_matrix.loc['bmi', 'systolic_bp']
    # print(f"Correlation between BMI and Systolic_BP: {correlation}")
    bp_correlations = correlation_matrix['systolic_bp'].sort_values(ascending=False)
    print("Correlations with Systolic_BP:")
    print(bp_correlations)
    return df


def phase_4_predictive_modeling(df):
    """
    [Phase 4] Predictive Analytics & Regression Modeling
    - Goal: Predict 'Systolic_BP' based on 'BMI'.
    - Train a Simple Linear Regression model.
    """
    print("\n[PHASE 4] Predictive Modeling ")
    print("-" * 65)
    # TODO: Set X = df[['BMI']] and y = df['Systolic_BP']
    # TODO: Fit Linear Regression model and output slope & intercept

    X = df[['bmi']]
    y = df['systolic_bp']

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_

    print(f"Regression Equation: Systolic_BP = {slope:.4f} * BMI + {intercept:.4f}")

    return model, X, y


def phase_5_evaluation_and_reporting(model, X, y, df):
    """
    [Phase 5] Evaluation & Prescriptive Recommendations (5 Marks)
    - Goal: Evaluate model performance using R-squared and MAE.
    - Provide prescriptive clinical recommendations based on findings.
    """
    print("\n[PHASE 5] Evaluation & Clinical Action (5 Marks)")
    print("-" * 65)
    # TODO: Calculate R2 score and Mean Absolute Error (MAE)
    # TODO: Print prescriptive clinical action items

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    print(f"Model R-squared (R2): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")

    if r2 < 0.5:
        print(
            "The model has limited explanatory power. "
            "Consider additional features and further validation."
        )
    else:
        print(
            "The model explains a substantial proportion of "
            "the observed variation in systolic BP. "
            "Further clinical validation is required."
        )

    print(
        "Clinical Note: BMI-based predictions should not replace "
        "actual BP measurements or professional medical assessment."
    )

def run_analytics_lifecycle():
    """
    Facade Orchestrator function.
    """
    print("=" * 65)
    print("STARTING HEALTHCARE ANALYTICS LIFECYCLE ")
    print("=" * 65)

    # Executing phases chronologically
    df = phase_1_discovery_and_setup()
    df = phase_2_descriptive_statistics(df)
    df = phase_3_bivariate_association(df)
    model, X, y = phase_4_predictive_modeling(df)
    phase_5_evaluation_and_reporting(model, X, y, df)


if __name__ == "__main__":
    run_analytics_lifecycle()