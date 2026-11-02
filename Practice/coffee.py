"""
Title: End-to-End Data Analytics Lifecycle
Description: A modular Python script demonstrating business discovery,
             data preparation, exploration, statistics, and predictive modeling.
"""

# ==========================================
# IMPORTS (Libraries for Data, Stats, & Viz)
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================
# SUB-STEP FUNCTIONS (The Lifecycle Phases)
# ==========================================

def phase_1_discovery_and_setup():
    """
    Module 1: Foundations, Business Understanding & Setup
    - Define business objective (e.g., Coffee Shop Revenue Analysis).
    - Load the 25-record real-world CSV dataset.
    - Initial inspection (head, info, missing values).
    """
    print("Executing Phase 1: Discovery & Data Setup...")

    df = pd.read_csv("coffee_shop_data.csv")
    print(df.head())
    print("Dataset Shape:", df.shape)
    df.info()

    print("Missing Values:\n", df.isnull().sum())
    return df


def phase_2_descriptive_statistics(df):
    """
    Module 2: Descriptive Statistics & Data Measures
    - Calculate Central Tendency (Mean, Median, Mode).
    - Calculate Dispersion (Standard Deviation, Variance, IQR, Range).
    - Check Distribution Shape & Outliers.
    """
    print("\nExecuting Phase 2: Descriptive Statistics & Measures...")

    # Standardize column names
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_")
    )

    mean = df['daily_revenue_usd'].mean()
    median = df['daily_revenue_usd'].median()
    mode = df['daily_revenue_usd'].mode()[0]
    std = df['daily_revenue_usd'].std()
    variance = df['daily_revenue_usd'].var()

    print("Mean:", mean)
    print("Median:", median)
    print("Mode:", mode)
    print("Standard Deviation:", std)
    print("Variance:", variance)

    Q1 = df['daily_revenue_usd'].quantile(0.25)
    Q3 = df['daily_revenue_usd'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print("Lower Bound:", lower_bound)
    print("Upper Bound:", upper_bound)
    print("IQR:", IQR)

    outliers = df[(df['daily_revenue_usd'] < lower_bound) | (df['daily_revenue_usd'] > upper_bound)]
    print("Outliers:\n", outliers)

    return df


def phase_3_bivariate_association(df):
    """
    Module 4: Bivariate & Multivariate Association
    - Analyze relationships between variables (e.g., Marketing Spend vs. Revenue).
    - Compute Covariance and Pearson Correlation Matrix.
    - Generate exploratory visualizations (Scatter plots, Correlation heatmap).
    """
    print("\nExecuting Phase 3: Bivariate Association & Visualization...")

    numerical_vals = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numerical_vals].corr(method='pearson')
    covariance_matrix = df[numerical_vals].cov()

    print("Correlation Matrix:\n", correlation_matrix)
    print("Covariance Matrix:\n", covariance_matrix)

    correlation = correlation_matrix.loc["marketing_spend_usd", "daily_revenue_usd"]
    print("Correlation between marketing and revenue:", correlation)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        df['marketing_spend_usd'],
        df['daily_revenue_usd'],
        alpha=0.8,
        marker='o',
        color='brown'
    )
    plt.xlabel("Marketing Spend (USD)")
    plt.ylabel("Revenue (USD)")
    plt.title("Correlation between Marketing Spend and Revenue")
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',
        fmt='.2f',
        linewidths=0.5
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    return df


def phase_4_predictive_modeling(df):
    """
    Module 6 & 8: Predictive Analytics & Regression Modeling
    - Set up a simple predictive model (e.g., Simple Linear Regression).
    - Train the model on independent variables to predict revenue.
    - Establish a baseline evaluation.
    """
    print("\nExecuting Phase 4: Predictive Modeling...")

    X = df[['marketing_spend_usd']]
    y = df['daily_revenue_usd']

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_

    print(f"Regression Equation: y = {slope:.4f}x + {intercept:.4f}")
    return model, X, y


def phase_5_evaluation_and_reporting(model, X, y, df):
    """
    Module 6 & 7: Model Evaluation, Generalization & Business Decision
    - Evaluate model performance (e.g., R-squared, Mean Absolute Error).
    - Discuss generalization and overfitting risks on small data.
    - Translate technical findings into prescriptive business recommendations.
    [Allocation: 5 Marks]
    """
    print("\nExecuting Phase 5: Evaluation & Business Action...")

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    print(f"Model R-squared (R2): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")

    print("\n--- Generalization & Overfitting Discussion ---")
    print("1. Sample Size Limitation: With only 25 records, the model is at high risk of overfitting.")
    print("2. Variance Risk: Coefficients may fluctuate significantly with new data; cross-validation or train-test splits are advised on larger samples.")

    print("\n--- Prescriptive Business Recommendations ---")
    print("1. Marketing Investment: The positive relationship confirms that increased marketing spend drives higher daily revenue.")
    print("2. Budget Optimization: Incrementally scale the marketing budget while tracking customer acquisition cost (CAC) and lifetime value (LTV).")
    print("3. Data Expansion: Collect a broader feature set (e.g., foot traffic, weather, day of week) and increase sample size for better predictive accuracy.")

    return r2, mae


# ==========================================
# THE FACADE FUNCTION (Orchestrator)
# ==========================================

def run_analytics_lifecycle():
    """
    Facade function that coordinates the entire data analytics
    lifecycle in chronological order.
    """
    print("=" * 60)
    print("STARTING COMPLETE DATA ANALYTICS LIFECYCLE")
    print("=" * 60)

    df = phase_1_discovery_and_setup()
    print("-" * 60)

    df = phase_2_descriptive_statistics(df)
    print("-" * 60)

    df = phase_3_bivariate_association(df)
    print("-" * 60)

    model, X, y = phase_4_predictive_modeling(df)
    print("-" * 60)

    phase_5_evaluation_and_reporting(model, X, y, df)

    print("=" * 60)
    print("ANALYTICS LIFECYCLE COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ==========================================
# MAIN GUARD
# ==========================================
if __name__ == "__main__":
    run_analytics_lifecycle()