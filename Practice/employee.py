"""
Title: Employee Productivity & Analytics Lifecycle
Description: Fill in the TODO sections across the 5 lifecycle phases.

"""


# Step 0: Import necessary libraries
# TODO: Import pandas, numpy, scikit-learn LinearRegression, and metrics
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery & Data Setup (5 Marks)
    - Goal: Load 'employee_productivity.csv' and inspect structure.
    - Check for null values and print summary info.
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)
    # TODO: Load CSV file using pandas
    # TODO: Print first 5 rows and verify shape
    # TODO: Check for missing values

    df = pd.read_csv("employee_productivity.csv")
    print("First 5 rows of the dataset:")
    print(df.head())
    print(f"Dataset shape: {df.shape}")
    print("Information about the dataset:")
    print(df.info())
    print("Missing values in each column:")
    print(df.isnull().sum())

    return df

def phase_2_descriptive_statistics(df):
    """
    [Phase 2] Descriptive Statistics & Data Measures
    - Goal: Compute measures of central tendency and dispersion
      for 'Weekly_Tasks_Completed'.
    - Check for potential outliers using IQR method.
    """
    print("\n[PHASE 2] Descriptive Statistics ")
    print("-" * 65)
    # TODO: Calculate Mean, Median, Std Dev, and IQR for Weekly_Tasks_Completed
    # TODO: Identify any values falling outside 1.5 * IQR bounds
    mean = df['Weekly_Tasks_Completed'].mean()
    median = df['Weekly_Tasks_Completed'].median()
    std_dev = df['Weekly_Tasks_Completed'].std()
    Q1 = df['Weekly_Tasks_Completed'].quantile(0.25)
    Q3 = df['Weekly_Tasks_Completed'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"Mean: {mean}, Median: {median}, Std Dev: {std_dev}")
    outliers = df[(df['Weekly_Tasks_Completed'] < lower_bound) | (df['Weekly_Tasks_Completed'] > upper_bound)]
    print(f"Outliers detected:\n{outliers}")

    return df


def phase_3_bivariate_association(df):
    """
    [Phase 3] Bivariate Association & Correlation
    - Goal: Examine relationships between Training Hours, Experience,
      and Tasks Completed.
    - Compute Pearson correlation matrix.
    """
    print("\n[PHASE 3] Bivariate Association ")
    print("-" * 65)
    # TODO: Select numeric columns and compute Pearson correlation matrix
    # TODO: Highlight the correlation between Training_Hours and Weekly_Tasks_Completed

    numeric_cols = df.select_dtypes(include=['number']).columns
    correlation_matrix = df[numeric_cols].corr(method='pearson')
    print("Pearson Correlation Matrix:")
    print(correlation_matrix)

    correlation = correlation_matrix.loc['Training_Hours', 'Weekly_Tasks_Completed']
    print(f"Correlation between Training_Hours and Weekly_Tasks_Completed: {correlation}")

    if correlation > 0:
        print("Positive correlation: More training hours are associated with more tasks completed.")
    elif correlation < 0:
        print("Negative correlation: More training hours are associated with fewer tasks completed.")
    else:
        print("No correlation between training hours and tasks completed.")

    return df

def phase_4_predictive_modeling(df):
    """
    [Phase 4] Predictive Analytics & Regression Modeling
    - Goal: Predict 'Weekly_Tasks_Completed' based on 'Training_Hours'.
    - Train a Simple Linear Regression model.
    """
    print("\n[PHASE 4] Predictive Modeling ")
    print("-" * 65)
    # TODO: Set X = df[['Training_Hours']] and y = df['Weekly_Tasks_Completed']
    # TODO: Fit Linear Regression model and output slope & intercept

    X = df[['Training_Hours']]
    y = df['Weekly_Tasks_Completed']

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_

    print(f"Linear Regression Model: Weekly_Tasks_Completed = {slope:.2f} * Training_Hours + {intercept:.2f}")

    return model, X, y



def phase_5_evaluation_and_reporting(model, X, y, df):
    """
    [Phase 5] Evaluation & Prescriptive Recommendation (5 Marks)
    - Goal: Evaluate model using R-squared and MAE.
    - Provide operational HR recommendations based on data.
    """
    print("\n[PHASE 5] Evaluation & HR Recommendations (5 Marks)")
    print("-" * 65)
    # TODO: Calculate R2 score and Mean Absolute Error (MAE)
    # TODO: Print prescriptive action item for employee training investments

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    print("R2 Score:", r2)
    print("Mean Absolute Error (MAE):", mae)

    correlation = df[['Training_Hours', 'Weekly_Tasks_Completed']].corr().iloc[0, 1]
    if correlation > 0.7 and r2 > 0.7:

        print("Strong relationship detected.")
        print("Invest in structured employee training.")

    elif correlation > 0.4 and r2 > 0.4:

        print("Moderate relationship detected.")
        print("Consider targeted training programs.")

    else:

        print("Limited predictive relationship detected.")
        print("Investigate other productivity factors.")

def run_analytics_lifecycle():
    """
    Facade Orchestrator function.
    """
    print("=" * 65)
    print("STARTING EMPLOYEE ANALYTICS LIFECYCLE (Out of 40 Marks)")
    print("=" * 65)

    # Executing phases chronologically
    df = phase_1_discovery_and_setup()
    df = phase_2_descriptive_statistics(df)
    df = phase_3_bivariate_association(df)
    model, X, y = phase_4_predictive_modeling(df)
    phase_5_evaluation_and_reporting(model, X, y, df)

    pd.set_option('display.max_columns', None)
    pd.set_option("display.width", None)
    print(df)


if __name__ == "__main__":
    run_analytics_lifecycle()