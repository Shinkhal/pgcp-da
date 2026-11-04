"""
Title: Food Delivery Time Analytics Lifecycle
Description: Fill in the TODO sections across the 5 lifecycle phases.
Dataset: delivery_times.csv  (90 rows)
Columns: Order_ID, City, Distance_KM, Traffic_Index (1-10), Items_Count, Delivery_Time_Min
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
    - Goal: Load 'delivery_times.csv' and inspect structure.
    - Check for null values and handle them (drop rows with missing values).
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)
    # TODO: Load CSV file using pandas
    # TODO: Print first 5 rows, shape, and df.info()
    # TODO: Count missing values per column and drop rows that contain missing values and print the new shape

    df = pd.read_csv('delivery_times.csv')
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(' ','_')
    )
    print(df.head())
    df.info()

    missing = df.isnull().sum()
    print(f"Missing values per column:\n{missing}")

    df = df.dropna()
    print(f"New shape after dropping rows with missing values: {df.shape}")

    return df


def phase_2_descriptive_statistics(df):
    """
    [Phase 2] Descriptive Statistics & Data Measures (10 Marks)
    - Goal: Compute central tendency and dispersion for 'Delivery_Time_Min'.
    - Detect outliers using the IQR method.
    """
    print("\n[PHASE 2] Descriptive Statistics")
    print("-" * 65)
    # TODO: Calculate Mean, Median, Std Dev, and IQR for Delivery_Time_Min
    # TODO: Compute lower/upper bounds (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    # TODO: Print the orders (Order_ID) whose Delivery_Time_Min falls outside the bounds
    # TODO: Print average Delivery_Time_Min per City (groupby)

    target = df['delivery_time_min']
    mean = target.mean()
    median = target.median()
    std_dev = target.std()
    print(f'Mean: {mean:.2f}, Median: {median:.2f}, Std Dev: {std_dev:.2f}')
    q1 = target.quantile(0.25)
    q3 = target.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(target < lower_bound) | (target > upper_bound)]
    print(f'Lower bound : {lower_bound:.2f}, Upper Bound : {upper_bound:.2f}')
    print(outliers)

    grouped_analytics = df.groupby('city')['delivery_time_min'].mean()
    print(grouped_analytics)

    return df


def phase_3_bivariate_association(df):
    """
    [Phase 3] Bivariate Association & Correlation (10 Marks)
    - Goal: Examine relationships among Distance_KM, Traffic_Index, Items_Count and Delivery_Time_Min.
    """
    print("\n[PHASE 3] Bivariate Association")
    print("-" * 65)
    # TODO: Compute Pearson correlation matrix of the numeric columns
    # TODO: Highlight corr(Distance_KM, Delivery_Time_Min) and interpret its strength
    # TODO: Name the feature most strongly correlated with Delivery_Time_Min

    numeric_cols = ['distance_km', 'traffic_index', 'items_count', 'delivery_time_min']
    correlation_matrix = df[numeric_cols].corr(method='pearson')
    print(correlation_matrix)

    corr_distance = correlation_matrix.loc['distance_km', 'delivery_time_min']
    print(f"Correlation between Distance_KM and Delivery_Time_Min: {corr_distance:.2f}")

    strongest_corr_feature = correlation_matrix['delivery_time_min'].drop('delivery_time_min').idxmax()
    print(f"The feature most strongly correlated with Delivery_Time_Min is: {strongest_corr_feature}")

    return df


def phase_4_predictive_modeling(df):
    """
    [Phase 4] Predictive Analytics & Regression Modeling (10 Marks)
    - Goal: Predict 'Delivery_Time_Min' from 'Distance_KM'.
    - Train a Simple Linear Regression model.
    """
    print("\n[PHASE 4] Predictive Modeling")
    print("-" * 65)
    # TODO: Set X = df[['Distance_KM']] and y = df['Delivery_Time_Min']
    # TODO: Fit LinearRegression; print slope and intercept
    # TODO: Predict Delivery_Time_Min when Distance_KM = 12 km and print the result
    # TODO: return model, X, y

    X = df[['distance_km']]
    y = df['delivery_time_min']

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_
    print(f"Linear Regression Formula: Delivery_Time_Min = {slope:.2f} * Distance_KM + {intercept:.2f}")

    new_data = pd.DataFrame({'distance_km': [12]})
    prediction = model.predict(new_data)
    print(f"Predicted Delivery_Time_Min for 12 km: {prediction[0]:.2f} minutes")

    return model, X, y


def phase_5_evaluation_and_reporting(model, X, y, df):
    """
    [Phase 5] Evaluation & Prescriptive Recommendation (5 Marks)
    - Goal: Evaluate the model using R-squared and MAE.
    - Provide operational recommendations for the logistics team.
    """
    print("\n[PHASE 5] Evaluation & Recommendations")
    print("-" * 65)
    # TODO: Calculate R2 score and Mean Absolute Error (MAE)
    # TODO: Print a prescriptive action item: estimate added minutes per extra km and recommend a maximum delivery radius to keep promised delivery times

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    print(f"Model R-squared (R2): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} minutes")

    if r2 < 0.5:
        print("The model has limited explanatory power. Consider additional features and further validation.")
    else:
        print(f"On average, each additional km adds approximately {model.coef_[0]:.2f} minutes to delivery time.")
        recommended_max_distance = 15  # Example threshold
        print(f"Recommendation: Limit delivery radius to {recommended_max_distance} km to maintain promised delivery times.")


def run_analytics_lifecycle():
    """Facade Orchestrator function."""
    print("=" * 65)
    print("STARTING DELIVERY ANALYTICS LIFECYCLE (Out of 40 Marks)")
    print("=" * 65)

    df = phase_1_discovery_and_setup()
    phase_2_descriptive_statistics(df)
    phase_3_bivariate_association(df)
    model, X, y = phase_4_predictive_modeling(df)
    phase_5_evaluation_and_reporting(model, X, y, df)


if __name__ == "__main__":
    run_analytics_lifecycle()
