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
from soupsieve.util import lower


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
    # TODO: Load dataset using pandas
    # TODO: Perform basic inspection (shape, data types, missing values)

    df = pd.read_csv("coffee_shop_data.csv")
    print(df.head())
    print(df.shape)
    df.info()

    print("missing Vals: " , df.isnull().sum())
    return df


def phase_2_descriptive_statistics(df):
    """
    Module 2: Descriptive Statistics & Data Measures
    - Calculate Central Tendency (Mean, Median, Mode).
    - Calculate Dispersion (Standard Deviation, Variance, IQR, Range).
    - Check Distribution Shape & Outliers.

    """
    print("Executing Phase 2: Descriptive Statistics & Measures...")
    # TODO: Compute summary statistics (mean, median, std, IQR)
    # TODO: Identify and handle any outliers

    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_")
    )

    mean = df['daily_revenue_usd'].mean()
    median = df['daily_revenue_usd'].median()
    mode = df['daily_revenue_usd'].mode()[0]
    std = df['daily_revenue_usd'].std()
    variance = df['daily_revenue_usd'].var()
    # iqr = df['daily_revenue_usd'].iqr()

    print("Mean",mean)
    print("Median",median)
    print("Mode",mode)
    print("Standard Deviation",std)
    print("Variance",variance)
    # print("IQR",iqr)


    Q1 = df['daily_revenue_usd'].quantile(0.25)
    Q3 = df['daily_revenue_usd'].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print("lower_bound",lower_bound)
    print("upper_bound",upper_bound)
    print("IQR",IQR)

    outliers = df[ (df['daily_revenue_usd'] < lower_bound) | (df['daily_revenue_usd'] > upper_bound)]

    print("outliers",outliers)

    return df


def phase_3_bivariate_association(df):
    """
    Module 4: Bivariate & Multivariate Association
    - Analyze relationships between variables (e.g., Marketing Spend vs. Revenue).
    - Compute Covariance and Pearson Correlation Matrix.
    - Generate exploratory visualizations (Scatter plots, Correlation heatmap).

    """
    print("Executing Phase 3: Bivariate Association & Visualization...")
    # TODO: Compute correlation matrix
    # TODO: Plot correlation heatmap and scatter plots

    numerical_vals = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numerical_vals].corr(method='pearson')
    covariance_matrix = correlation_matrix.cov()
    print("Correlation Matrix",correlation_matrix)
    print("Correlation Variance",covariance_matrix)

    correlation = correlation_matrix.loc["marketing_spend_usd","daily_revenue_usd"]
    print("Correlation between marketing and revenue",correlation)

    plt.figure(figsize=(10,10))
    plt.scatter(
        df['marketing_spend_usd'],
        df['daily_revenue_usd'],
        alpha=0.8,
        marker='o'
    )
    plt.xlabel("Marketing Spend (USD)")
    plt.ylabel("Revenue (USD)")
    plt.title("Correlation between marketing and revenue")
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    return df


def phase_4_predictive_modeling():
    """
    Module 6 & 8: Predictive Analytics & Regression Modeling
    - Set up a simple predictive model (e.g., Simple Linear Regression).
    - Train the model on independent variables to predict revenue.
    - Establish a baseline evaluation.

    """
    print("Executing Phase 4: Predictive Modeling...")
    # TODO: Define features (X) and target (y)
    # TODO: Fit a simple linear regression model
    # TODO: Extract model coefficients and intercept

    X = df[['marketing_spend_usd']]
    y = df['daily_revenue_usd']

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_

    print("Slope",slope,"Intercept",intercept)
    return model, X, y
def phase_5_evaluation_and_reporting(model, X, y):
    """
    Module 6 & 7: Model Evaluation, Generalization & Business Decision
    - Evaluate model performance (e.g., R-squared, Mean Absolute Error).
    - Discuss generalization and overfitting risks on small data.
    - Translate technical findings into prescriptive business recommendations.
    [Allocation: 5 Marks]
    """
    print("Executing Phase 5: Evaluation & Business Action...")
    # TODO: Calculate evaluation metrics
    # TODO: Print final business recommendations based on findings
    pass


# ==========================================
# THE FACADE FUNCTION (Orchestrator)
# ==========================================

def run_analytics_lifecycle():
    """
    Facade function that coordinates the entire data analytics
    lifecycle in chronological order.
    """
    print("=" * 60)
    print("STARTING COMPLETE DATA ANALYTICS LIFECYCLE (Out of 40 Marks)")
    print("=" * 60)

    df=phase_1_discovery_and_setup()
    print("-" * 60)

    df=phase_2_descriptive_statistics(df)
    print("-" * 60)

    df=phase_3_bivariate_association(df)
    print("-" * 60)

    model,X,y = phase_4_predictive_modeling(df)
    print("-" * 60)

    phase_5_evaluation_and_reporting(model, X, y)

    print("=" * 60)
    print("ANALYTICS LIFECYCLE COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ==========================================
# MAIN GUARD
# ==========================================
if __name__ == "__main__":
    run_analytics_lifecycle()