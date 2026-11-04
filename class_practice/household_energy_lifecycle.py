"""
Title: Household Energy Consumption Analytics Lifecycle
Description: Fill in the TODO sections across the 5 lifecycle phases.
Dataset: household_energy.csv  (65 rows)
Columns: Household_ID, Zone, Avg_Temp_C, Family_Size, Appliance_Count, Monthly_kWh
"""
from statistics import correlation

# Step 0: Import necessary libraries
# TODO: Import pandas, numpy, scikit-learn LinearRegression, and metrics
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.model_selection import train_test_split

def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery & Data Setup (5 Marks)
    - Goal: Load 'household_energy.csv' and inspect structure.
    - Check for null values and handle them (fill numeric NaNs with the mean).
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)
    # TODO: Load CSV file using pandas
    # TODO: Print first 5 rows, shape, and df.info()
    # TODO: Count missing values per column and impute numeric NaNs with the column mean

    df = pd.read_csv('household_energy.csv')

    print("Head: ",df.head())
    print("Columns: ",df.columns)
    print("-" * 65)
    print("shape" , df.shape)

    df.info()

    missing_vals = df.isnull().sum()
    print("Missing Values: ",missing_vals)

    numeric_cols = df.select_dtypes(include=np.number)
    for col in numeric_cols:
        mean_val = df[col].mean()
        df[col]=df[col].fillna(mean_val)
    # print(numeric_cols)
    # print(df.isnull().sum())
    return df


def phase_2_descriptive_statistics(df):
    """
    [Phase 2] Descriptive Statistics & Data Measures (10 Marks)
    - Goal: Compute central tendency and dispersion for 'Monthly_kWh'.
    - Detect outliers using the IQR method.
    """
    print("\n[PHASE 2] Descriptive Statistics")
    print("-" * 65)
    # TODO: Calculate Mean, Median, Std Dev, and IQR for Monthly_kWh
    # TODO: Compute lower/upper bounds (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    # TODO: Print the households (Household_ID) whose Monthly_kWh falls outside the bounds
    # TODO: Print average Monthly_kWh per Zone (groupby)

    df.columns =df.columns.str.strip().str.lower()
    mean = df['monthly_kwh'].mean()
    std = df['monthly_kwh'].std()
    median = df['monthly_kwh'].median()

    q1=df['monthly_kwh'].quantile(0.25)
    q3=df['monthly_kwh'].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr

    print("Mean: ",mean)
    print("Median: ",median)
    print("Lower bound: ",lower_bound)
    print("Upper bound: ",upper_bound)
    print("IQR: ",iqr)
    print("Std Deviation: ",std)
    outlliers = df[(df['monthly_kwh'] < lower_bound) | (df['monthly_kwh'] > upper_bound)]
    print("Outliers: ",outlliers)
    print(outlliers['household_id'])
    df=df[(df['monthly_kwh'] >=lower_bound) & (df['monthly_kwh'] <= upper_bound)]
    average = df.groupby('zone')['monthly_kwh'].mean()
    print("Average: ",average)
    print(df.shape)
    return df

def phase_3_bivariate_association(df):
    """
    [Phase 3] Bivariate Association & Correlation (10 Marks)
    - Goal: Examine relationships among Avg_Temp_C, Family_Size, Appliance_Count and Monthly_kWh.
    """
    print("\n[PHASE 3] Bivariate Association")
    print("-" * 65)
    # TODO: Compute Pearson correlation matrix of the numeric columns
    # TODO: Highlight corr(Avg_Temp_C, Monthly_kWh) and interpret its strength
    # TODO: Name the feature most strongly correlated with Monthly_kWh

    numeric_cols = df.select_dtypes(include=np.number)
    correlation_matrix = numeric_cols.corr(method='pearson')
    print("Correlation: ",correlation_matrix)

    correlation = correlation_matrix.loc["avg_temp_c","monthly_kwh"]
    print("Correlation: ",correlation)

    most_strong = correlation_matrix['monthly_kwh'].drop('monthly_kwh').idxmax()
    print("Most strong: ",most_strong)

    return df

def phase_4_predictive_modeling(df):
    """
    [Phase 4] Predictive Analytics & Regression Modeling (10 Marks)
    - Goal: Predict 'Monthly_kWh' from 'Avg_Temp_C'.
    - Train a Simple Linear Regression model.
    """
    print("\n[PHASE 4] Predictive Modeling")
    print("-" * 65)
    # TODO: Set X = df[['Avg_Temp_C']] and y = df['Monthly_kWh']
    # TODO: Fit LinearRegression; print slope and intercept
    # TODO: Predict Monthly_kWh when Avg_Temp_C = 35 C and print the result
    # TODO: model, X_train , x_test , y_train , y_test

    X=df[["avg_temp_c"]]
    y=df["monthly_kwh"]

    X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

    model = LinearRegression()
    model.fit(X_train,y_train)

    slope = model.coef_[0]
    intercept = model.intercept_

    print("Slope: ",slope)
    print("Intercept: ",intercept)

    new_data = pd.DataFrame({
        "avg_temp_c":[35]
    })

    prediction = model.predict(new_data)

    print("Prediction: ",prediction[0])

    return model , X_train,x_test,y_train,y_test




def phase_5_evaluation_and_reporting(model, X_train , x_test , y_train , y_test , df):
    """
    [Phase 5] Evaluation & Prescriptive Recommendation (5 Marks)
    - Goal: Evaluate the model using R-squared and MAE.
    - Provide operational recommendations for the electricity utility's demand-planning team.
    """
    print("\n[PHASE 5] Evaluation & Recommendations")
    print("-" * 65)
    # TODO: Calculate R2 score and Mean Absolute Error (MAE)
    # TODO: Print a prescriptive action item: estimate extra kWh per +1 C and recommend peak-load capacity planning or tariff incentives for hot months

    y_pred = model.predict(x_test)
    r2=r2_score(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)

    print("R-squared: ",r2)
    print("MAE: ",mae)


def run_analytics_lifecycle():
    """Facade Orchestrator function."""
    print("=" * 65)
    print("STARTING ENERGY ANALYTICS LIFECYCLE (Out of 40 Marks)")
    print("=" * 65)

    df = phase_1_discovery_and_setup()
    phase_2_descriptive_statistics(df)
    phase_3_bivariate_association(df)
    model, X_train , x_test , y_train , y_test = phase_4_predictive_modeling(df)
    phase_5_evaluation_and_reporting(model, X_train , x_test , y_train , y_test, df)


if __name__ == "__main__":
    run_analytics_lifecycle()
