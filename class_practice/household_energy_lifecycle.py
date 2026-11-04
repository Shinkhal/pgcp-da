"""
Title: Household Energy Consumption Analytics Lifecycle
Description: Fill in the TODO sections across the 5 lifecycle phases.
Dataset: household_energy.csv  (65 rows)
Columns: Household_ID, Zone, Avg_Temp_C, Family_Size, Appliance_Count, Monthly_kWh
"""

# Step 0: Import necessary libraries
# TODO: Import pandas, numpy, scikit-learn LinearRegression, and metrics


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
    pass


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
    pass


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
    pass


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
    # TODO: return model, X, y
    pass


def phase_5_evaluation_and_reporting(model, X, y, df):
    """
    [Phase 5] Evaluation & Prescriptive Recommendation (5 Marks)
    - Goal: Evaluate the model using R-squared and MAE.
    - Provide operational recommendations for the electricity utility's demand-planning team.
    """
    print("\n[PHASE 5] Evaluation & Recommendations")
    print("-" * 65)
    # TODO: Calculate R2 score and Mean Absolute Error (MAE)
    # TODO: Print a prescriptive action item: estimate extra kWh per +1 C and recommend peak-load capacity planning or tariff incentives for hot months
    pass


def run_analytics_lifecycle():
    """Facade Orchestrator function."""
    print("=" * 65)
    print("STARTING ENERGY ANALYTICS LIFECYCLE (Out of 40 Marks)")
    print("=" * 65)

    df = phase_1_discovery_and_setup()
    # phase_2_descriptive_statistics(df)
    # phase_3_bivariate_association(df)
    # model, X, y = phase_4_predictive_modeling(df)
    # phase_5_evaluation_and_reporting(model, X, y, df)


if __name__ == "__main__":
    run_analytics_lifecycle()
